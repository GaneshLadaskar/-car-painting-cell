"""
RoboDK script - robotic car painting cell.

What it does, in order
  1. BUILD    (first run only) loads the car, floor, conveyor, rooms, fence, robot pedestal
              and the spray gun, creates the CarFrame reference frame
  2. PLAN     for every robot position around the car, tests with inverse kinematics which
              painting strokes the robot can reach, and assigns each stroke to one position
  3. PAINT    drives the robot around the car, paints the assigned strokes with the spray
              simulation, and checks for collisions after every waypoint
  4. REPORT   writes results/run_log.csv and results/summary.json

How to run
  - Open RoboDK, make sure ONE robot is in the station (File > Open online library).
  - Set PROJECT_DIR below.
  - Run this file from RoboDK (right-click the Python script > Run) or from VS Code.
  - First try ANALYZE_ONLY = True: no painting, just prints how much of the car is reachable.
"""
import csv
import json
import math
import os
import time

from robodk import robolink, robomath
from robodk.robolink import *      # constants such as ITEM_TYPE_ROBOT, COLLISION_ON, SPRAY_ON
from robodk.robomath import *      # Mat, transl, rotz, invH

# =============================================================== settings
PROJECT_DIR = r"C:\car-painting-cell"     # folder that contains models/ paths/ results/
ROBOT_NAME = ""                           # empty = the only robot in the station
ANALYZE_ONLY = False                      # True = just check reachability, do not paint

PAINT_SPEED = 500.0         # mm/s along the stroke
JOINT_SPEED = 90.0          # deg/s for repositioning moves
APPROACH_MM = 150.0         # extra distance backed off from the stroke before/after spraying
JOINT_JUMP_DEG = 25.0       # a stroke is rejected if any joint jumps more than this between two waypoints
LINK_DIST_MM = 700.0        # strokes closer than this are linked with a straight move, else a joint move
MIN_SEG_POINTS = 3          # ignore reachable fragments shorter than this many waypoints

# Robot positions around the car (the robot base is driven to these; think of a carriage on a rail).
# name, x [mm], y [mm], yaw [deg]  -  yaw makes the robot face the car. Lap runs counter-clockwise.
SIDE_Y = 1800.0             # distance from the car centre line to the robot axis (side positions)
END_X = 3700.0              # distance from the car centre to the robot axis (front / rear positions)
STATIONS = [
    ("right_rear", -1700, -SIDE_Y, 90),
    ("right_mid", 0, -SIDE_Y, 90),
    ("right_front", 1700, -SIDE_Y, 90),
    ("front", END_X, 0, 180),
    ("left_front", 1700, SIDE_Y, 270),
    ("left_mid", 0, SIDE_Y, 270),
    ("left_rear", -1700, SIDE_Y, 270),
    ("rear", -END_X, 0, 360),
]
# Corner waypoints (x, y, yaw) the base passes through AFTER leaving the named station,
# so the robot base does not cut across the corners of the car.
VIA_POINTS = {
    "right_front": [(END_X, -SIDE_Y, 135)],
    "front": [(END_X, SIDE_Y, 225)],
    "left_rear": [(-END_X, SIDE_Y, 315)],
}

BASE_STEP_MM = 80.0         # animation resolution while the base moves
BASE_STEP_DEG = 4.0

CAR_COLOR = [0.82, 0.84, 0.88, 1.0]
PAINT_RGB = [0.78, 0.05, 0.05]          # used for every surface when USE_ZONE_COLORS is False
USE_ZONE_COLORS = True                  # one colour per surface: makes the demo easy to follow
ZONE_RGB = {
    "hood": [0.80, 0.08, 0.08], "roof": [0.10, 0.35, 0.85], "trunk": [0.95, 0.65, 0.05],
    "right_side": [0.10, 0.60, 0.30], "left_side": [0.55, 0.20, 0.75],
    "front": [0.05, 0.65, 0.70], "rear": [0.90, 0.40, 0.10],
}

# =============================================================== helpers
def mat_to_list(m):
    """Flatten a joints Mat (Nx1 or 1xN) to a plain list. Empty list when there is no solution."""
    r, c = m.size()
    if r == 0 or c == 0:
        return []
    return [m.rows[i][j] for i in range(r) for j in range(c)]


def pose_from_flat(flat):
    return Mat([flat[0:4], flat[4:8], flat[8:12], flat[12:16]])


def base_pose(x, y, yaw_deg, ped_h):
    return transl(x, y, ped_h) * rotz(math.radians(yaw_deg))


def log(msg):
    print(msg)


# =============================================================== stage 1: build
def build_cell(RDK, robot, cfg, models):
    log("building the cell ...")
    ped_h, conv_h, gun_len = cfg["ped_h"], cfg["conv_h"], cfg["gun_len"]

    def load(name, parent=None, color=None, item_name=None):
        item = RDK.AddFile(os.path.join(models, name + ".stl"), parent) if parent else \
            RDK.AddFile(os.path.join(models, name + ".stl"))
        item.setName(item_name or name)
        if color:
            item.setColor(color)
        return item

    car_frame = RDK.AddFrame("CarFrame")
    car_frame.setPose(transl(0, 0, conv_h + cfg.get("gap", 2.0)))   # car rests on the conveyor
    load("car", car_frame, CAR_COLOR, "Car")

    load("floor", None, [0.55, 0.57, 0.60, 1.0])
    load("conveyor", None, [0.25, 0.25, 0.28, 1.0])
    load("control_rooms", None, [0.70, 0.74, 0.80, 1.0])
    load("fence", None, [0.95, 0.80, 0.10, 0.35])

    # robot base frame: the robot's parent frame in the tree, so moving it moves the robot
    base_frame = robot.Parent()
    if base_frame.Type() != ITEM_TYPE_FRAME:
        base_frame = RDK.AddFrame("RobotBase")
        robot.setParent(base_frame)
    name, x, y, yaw = STATIONS[0]
    base_frame.setPose(base_pose(x, y, yaw, ped_h))
    load("pedestal", base_frame, [0.35, 0.35, 0.38, 1.0], "Pedestal")

    # spray gun: an object added to a robot is turned into a tool automatically
    gun = RDK.AddFile(os.path.join(models, "gun.stl"), robot)
    gun.setName("SprayGun")
    gun.setColor([0.15, 0.15, 0.18, 1.0])
    gun.setPoseTool(transl(0, 0, gun_len))                  # TCP at the nozzle tip, Z = spray direction
    robot.setPoseTool(gun)
    log("cell built")


# =============================================================== stage 2/3: plan and paint
class Painter:
    def __init__(self, RDK, robot, car_frame, car, tool, base_frame, cfg, paths):
        self.RDK, self.robot, self.car_frame, self.car, self.tool = RDK, robot, car_frame, car, tool
        self.base_frame, self.ped_h = base_frame, cfg["ped_h"]
        self.n_axes = len(mat_to_list(robot.Joints()))
        self.home = robot.JointsHome()
        self.standoff = paths["meta"]["standoff_mm"]
        self.pending = []                    # stroke pieces nobody has reached yet
        self.assigned = []                   # stroke pieces given to a robot position
        self.total_points = {}
        for zone in paths["meta"]["zone_order"]:
            self.total_points[zone] = 0
            for k, stroke in enumerate(paths["zones"][zone]):
                poses = [pose_from_flat(p) for p in stroke]
                self.total_points[zone] += len(poses)
                self.pending.append(self._piece(zone, "%s_%02d" % (zone, k), poses, None))
        self.dropped_points = 0
        self.tool_pose = self.frame_pose = None
        self.collision_events = []
        self.spray_ids = {}
        self.limits = self._joint_limits()

    # ---------------------------------------------------------- kinematics helpers
    def _joint_limits(self):
        try:
            lo, hi, _ = self.robot.JointLimits()
            lo, hi = mat_to_list(lo), mat_to_list(hi)
            if len(lo) >= self.n_axes and len(hi) >= self.n_axes:
                return lo[:self.n_axes], hi[:self.n_axes]
        except Exception:
            pass
        return None

    def _within_limits(self, j):
        if not self.limits:
            return True
        lo, hi = self.limits
        return all(lo[i] - 0.5 <= j[i] <= hi[i] + 0.5 for i in range(self.n_axes))

    @staticmethod
    def _piece(zone, pid, poses, station):
        return {"zone": zone, "id": pid, "poses": poses, "station": station,
                "status": "planned" if station else "unreachable", "collisions": 0}

    def refresh_frames(self):
        """Pose of the car frame as seen from the robot base, and the TCP offset (both needed for IK).
        Must be refreshed whenever the robot base has moved."""
        self.tool_pose = self.robot.PoseTool()
        self.frame_pose = invH(self.robot.PoseAbs()) * self.car_frame.PoseAbs()

    @staticmethod
    def approach(pose):
        return pose * transl(0, 0, -APPROACH_MM)       # back off along the tool Z axis

    def _ik(self, pose, seed=None):
        sol = mat_to_list(self.robot.SolveIK(pose, seed, self.tool_pose, self.frame_pose))[:self.n_axes]
        if len(sol) == self.n_axes and self._within_limits(sol):
            return sol
        return None

    def reach_segments(self, poses):
        """Index ranges (a, b) of consecutive waypoints the robot can follow from here,
        i.e. an IK solution exists, it is inside the joint limits and no joint jumps."""
        segs, start, prev = [], None, None
        for i, pose in enumerate(poses):
            sol = self._ik(pose, prev)
            if sol is None:
                if start is not None:
                    segs.append((start, i - 1))
                start = None
            elif prev is not None and max(abs(a - b) for a, b in zip(sol, prev)) > JOINT_JUMP_DEG:
                segs.append((start, i - 1))          # joint jump: end the segment, begin a new one here
                start = i
            elif start is None:
                start = i
            prev = sol
        if start is not None:
            segs.append((start, len(poses) - 1))
        return [(a, b) for a, b in segs if b - a + 1 >= MIN_SEG_POINTS]

    def plan_station(self, name):
        """Take every pending stroke, give this station the parts it can reach, keep the rest pending."""
        self.refresh_frames()
        taken, still_pending = [], []
        for run in self.pending:
            poses = run["poses"]
            segs = [(a, b) for (a, b) in self.reach_segments(poses)
                    if self._ik(self.approach(poses[a])) and self._ik(self.approach(poses[b]))]
            covered = [False] * len(poses)
            for (a, b) in segs:
                taken.append(self._piece(run["zone"], "%s_%d-%d" % (run["id"], a, b), poses[a:b + 1], name))
                for i in range(a, b + 1):
                    covered[i] = True
            i = 0
            while i < len(poses):                    # leftovers wait for a later station
                if covered[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(poses) and not covered[j + 1]:
                    j += 1
                if j - i + 1 >= MIN_SEG_POINTS:
                    still_pending.append(self._piece(run["zone"], "%s_%d-%d" % (run["id"], i, j), poses[i:j + 1], None))
                else:
                    self.dropped_points += j - i + 1
                i = j + 1
        self.pending = still_pending
        self.assigned.extend(taken)
        return taken

    # ---------------------------------------------------------- base movement
    def goto_home_pose(self):
        self.robot.setSpeed(-1, JOINT_SPEED)
        self.robot.MoveJ(self.home)

    def move_base(self, frm, to):
        """Animate the robot base from (x, y, yaw) to (x, y, yaw)."""
        dist = math.hypot(to[0] - frm[0], to[1] - frm[1])
        steps = max(1, int(max(dist / BASE_STEP_MM, abs(to[2] - frm[2]) / BASE_STEP_DEG)))
        for i in range(1, steps + 1):
            t = i / float(steps)
            x = frm[0] + (to[0] - frm[0]) * t
            y = frm[1] + (to[1] - frm[1]) * t
            yaw = frm[2] + (to[2] - frm[2]) * t
            self.base_frame.setPose(base_pose(x, y, yaw, self.ped_h))
            if self.RDK.Collisions() > 0:
                self.collision_events.append({"where": "base_move", "x": round(x), "y": round(y)})

    # ---------------------------------------------------------- spray
    def setup_spray(self):
        """One simulated spray gun per surface (same nozzle, different colour)."""
        S = self.standoff
        options = "ELLYPSE PROJECT PARTICLE=SPHERE(4,8,1,1,0.5) STEP=8x8 RAND=2"
        self.RDK.Spray_Clear()
        for zone in self.total_points:
            rgb = ZONE_RGB.get(zone, PAINT_RGB) if USE_ZONE_COLORS else PAINT_RGB
            # volume is given relative to the TCP (nozzle tip), Z = spray direction:
            # a tiny ellipse at the nozzle opening widening to 100 x 300 mm at the surface
            close_p = [0, 0, 0] + [4, 0, 0] + [0, 4, 0] + rgb + [1.0]
            far_p = [0, 0, S + 40] + [50, 0, S + 40] + [0, 150, S + 40] + rgb + [0.7]
            volume = Mat([close_p, far_p]).tr()
            self.spray_ids[zone] = self.RDK.Spray_Add(self.tool, self.car, options, volume)
            self.RDK.Spray_SetState(SPRAY_OFF, self.spray_ids[zone])

    def spray(self, on, zone):
        self.RDK.Spray_SetState(SPRAY_ON if on else SPRAY_OFF, self.spray_ids[zone])

    # ---------------------------------------------------------- painting a stroke
    def check_collision(self, run, label):
        n = self.RDK.Collisions()
        if n > 0:
            run["collisions"] += 1
            self.collision_events.append({"run": run["id"], "at": label, "pairs": n})
        return n

    def paint_run(self, run, last_end):
        robot = self.robot
        first, last = run["poses"][0], run["poses"][-1]
        start = self.approach(first)
        try:
            robot.setSpeed(PAINT_SPEED, JOINT_SPEED)
            if last_end is not None and (last_end.Pos()[0] - start.Pos()[0]) ** 2 + \
                    (last_end.Pos()[1] - start.Pos()[1]) ** 2 + (last_end.Pos()[2] - start.Pos()[2]) ** 2 \
                    < LINK_DIST_MM ** 2:
                robot.MoveL(start)
            else:
                robot.MoveJ(start)
            self.check_collision(run, "approach")
            robot.MoveL(first)
            self.spray(True, run["zone"])
            for i, pose in enumerate(run["poses"][1:], start=1):
                robot.MoveL(pose)
                self.check_collision(run, "waypoint %d" % i)
            self.spray(False, run["zone"])
            end = self.approach(last)
            robot.MoveL(end)
            run["status"] = "painted" if run["collisions"] == 0 else "painted_with_collision_flags"
            return end
        except Exception as exc:
            self.spray(False, run["zone"])
            run["status"] = "failed: " + str(exc).splitlines()[0][:80]
            try:
                self.goto_home_pose()
            except Exception:
                pass
            return None

    # ---------------------------------------------------------- collision self-test
    def collision_selftest(self, run):
        """Put the gun tip 20 mm INSIDE the car and ask RoboDK whether it notices."""
        bad = run["poses"][len(run["poses"]) // 2] * transl(0, 0, self.standoff + 20)
        sol = self._ik(bad)
        if sol is None:
            return None
        self.robot.setJoints(sol)
        n = self.RDK.Collisions()
        self.goto_home_pose()
        return n


def main():
    RDK = Robolink()
    RDK.setRunMode(RUNMODE_SIMULATE)
    models = os.path.join(PROJECT_DIR, "models")
    results = os.path.join(PROJECT_DIR, "results")
    os.makedirs(results, exist_ok=True)

    with open(os.path.join(models, "cell_config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(PROJECT_DIR, "paths", "paths.json")) as f:
        paths = json.load(f)

    robot = RDK.Item(ROBOT_NAME, ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise Exception("No robot found in the station. Add one from the RoboDK library first.")

    if not RDK.Item("CarFrame", ITEM_TYPE_FRAME).Valid():
        build_cell(RDK, robot, cfg, models)

    car_frame = RDK.Item("CarFrame", ITEM_TYPE_FRAME)
    car = RDK.Item("Car", ITEM_TYPE_OBJECT)
    tool = RDK.Item("SprayGun", ITEM_TYPE_TOOL)
    if not (car.Valid() and tool.Valid()):
        raise Exception("Could not find 'Car' / 'SprayGun' in the station tree.")

    robot.setPoseFrame(car_frame)
    robot.setPoseTool(tool)
    RDK.setCollisionActive(COLLISION_ON)

    painter = Painter(RDK, robot, car_frame, car, tool, robot.Parent(), cfg, paths)
    painter.goto_home_pose()
    painter.setup_spray()
    if RDK.Collisions() > 0:
        log("WARNING: the station is already in collision before painting starts. "
            "Check Tools > Collision map and look for overlapping objects (shown in red).")
    t0 = time.time()

    # ---- walk the lap: at each station take the stroke pieces it can reach, then paint them
    current = STATIONS[0][1:]
    painter.base_frame.setPose(base_pose(current[0], current[1], current[2], cfg["ped_h"]))
    selftest = None
    for si, (name, x, y, yaw) in enumerate(STATIONS):
        painter.move_base(current, (x, y, yaw))
        current = (x, y, yaw)

        taken = painter.plan_station(name)
        pts = sum(len(r["poses"]) for r in taken)
        left = sum(len(r["poses"]) for r in painter.pending)
        log("%-12s takes %4d waypoints in %3d pieces   (%d waypoints still unreached)" % (name, pts, len(taken), left))

        if selftest is None and taken:
            selftest = painter.collision_selftest(taken[0])
            log("collision self-test (gun pushed 20 mm into the body): %s collision pair(s) detected" % selftest)

        if not ANALYZE_ONLY:
            last_end = None
            for r in taken:
                last_end = painter.paint_run(r, last_end)
            painter.goto_home_pose()

        # drive to the next station through the corner via-point, if there is one
        if si + 1 < len(STATIONS):
            for via in VIA_POINTS.get(name, []):
                painter.move_base(current, via)
                current = via

    # ---- report
    all_points = sum(painter.total_points.values())
    planned_points = sum(len(r["poses"]) for r in painter.assigned)
    zones = {}
    for z, tot in painter.total_points.items():
        zones[z] = {"waypoints": tot, "planned": 0, "painted": 0}
    for r in painter.assigned:
        zones[r["zone"]]["planned"] += len(r["poses"])
        if r["status"].startswith("painted"):
            zones[r["zone"]]["painted"] += len(r["poses"])

    summary = {
        "waypoints_total": all_points,
        "waypoints_reachable": planned_points,
        "coverage_percent": round(100.0 * planned_points / max(all_points, 1), 1),
        "stroke_pieces": len(painter.assigned),
        "collision_events": len(painter.collision_events),
        "collision_selftest_pairs": selftest,
        "analyze_only": ANALYZE_ONLY,
        "elapsed_s": round(time.time() - t0, 1),
        "zones": zones,
    }
    with open(os.path.join(results, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(results, "run_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["piece", "zone", "station", "waypoints", "status", "collision_flags"])
        for r in painter.assigned + painter.pending:
            w.writerow([r["id"], r["zone"], r["station"] or "-", len(r["poses"]), r["status"], r["collisions"]])
    with open(os.path.join(results, "collision_events.json"), "w") as f:
        json.dump(painter.collision_events, f, indent=2)

    log("\n==== summary ====")
    log("reachable waypoints : %d / %d  (%.1f %%)" % (planned_points, all_points, summary["coverage_percent"]))
    for z, v in zones.items():
        log("  %-11s %4d / %4d waypoints reachable, %4d painted" % (z, v["planned"], v["waypoints"], v["painted"]))
    log("collision events    : %d" % len(painter.collision_events))
    RDK.ShowMessage("Done: %.1f %% of waypoints reachable, %d collision events" %
                    (summary["coverage_percent"], len(painter.collision_events)), False)


if __name__ == "__main__":
    main()

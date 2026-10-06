"""
Generate the painting trajectories for the car.

Idea
  The car surface is sampled with rays (ray casting).  For every zone of the car we
  lay out parallel raster strokes, find where each ray hits the body, read the
  surface normal there, and place the spray gun `standoff` mm above the surface with
  its Z axis pointing straight at the paint.  Stroke direction becomes the tool X axis.

Zones  (car frame: X = front, Y = left, Z = up, origin at the middle of the car on the ground)
    hood, roof, trunk     top-facing surfaces, strokes run across the car (along Y)
    right_side, left_side side panels,         strokes run vertically
    front, rear           end panels,          strokes run vertically

Output
    paths/paths.json          poses (4x4, row-major, mm) for every stroke, in the car frame
    paths/paths_preview.png   top / side / front / 3D view for your report

Usage
    python tools/generate_paths.py
    python tools/generate_paths.py --standoff 300 --pass-spacing 200 --step 60
"""
import argparse
import json
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ZONE_ORDER = ["hood", "roof", "trunk", "right_side", "left_side", "front", "rear"]
ZONE_COLORS = {"hood": "#e6194b", "roof": "#3cb44b", "trunk": "#4363d8", "right_side": "#f58231",
               "left_side": "#911eb4", "front": "#008080", "rear": "#9a6324"}


# ------------------------------------------------------------------ geometry helpers
def cast_rays(mesh, origins, direction):
    """Shoot one ray per origin along `direction`. Returns hit points and surface normals
    (NaN where the ray missed). Normals are flipped so they face the ray origin."""
    n = len(origins)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    dirs = np.tile(d, (n, 1))
    locs, ray_idx, tri_idx = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)

    hits = np.full((n, 3), np.nan)
    normals = np.full((n, 3), np.nan)
    best = {}
    for k, r in enumerate(ray_idx):               # keep the closest hit per ray
        dist = np.linalg.norm(locs[k] - origins[r])
        if r not in best or dist < best[r][0]:
            best[r] = (dist, k)
    for r, (_, k) in best.items():
        nrm = mesh.face_normals[tri_idx[k]].copy()
        if np.dot(nrm, d) > 0:
            nrm = -nrm
        hits[r] = locs[k]
        normals[r] = nrm
    return hits, normals


def split_runs(valid, hits, max_gap):
    """Indices of consecutive valid hits, broken wherever a hit is missing or the surface jumps."""
    runs, current = [], []
    for i in range(len(valid)):
        if not valid[i]:
            if current:
                runs.append(current)
            current = []
            continue
        if current and np.linalg.norm(hits[i] - hits[current[-1]]) > max_gap:
            runs.append(current)
            current = []
        current.append(i)
    if current:
        runs.append(current)
    return runs


def smooth_normals(normals, window=5):
    """Moving average along a stroke so the gun orientation does not jitter on faceted meshes."""
    if len(normals) < 3:
        return normals
    pad = window // 2
    padded = np.pad(normals, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    out = np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(3)], axis=1)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def tool_poses(points, normals, standoff):
    """Spray-gun pose for each point of a stroke (4x4 matrices).
         position : surface point + normal * standoff
         Z axis   : points into the surface (spray direction)
         X axis   : direction of travel
         Y axis   : Z x X
    """
    normals = smooth_normals(normals)
    poses = []
    for i in range(len(points)):
        z = -normals[i]
        a, b = max(i - 1, 0), min(i + 1, len(points) - 1)
        t = points[b] - points[a]
        t = t - np.dot(t, z) * z                  # keep travel direction in the tool plane
        if np.linalg.norm(t) < 1e-6:
            t = np.cross(z, [0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0])
        x = t / np.linalg.norm(t)
        y = np.cross(z, x)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
        T[:3, 3] = points[i] + normals[i] * standoff
        poses.append(T)
    return poses


# ------------------------------------------------------------------ stroke builders
def build_strokes(mesh, args):
    lo, hi = mesh.bounds
    margin = args.margin
    max_gap = 3.0 * max(args.step, 1.0) + 30.0
    strokes = {z: [] for z in ZONE_ORDER}

    def add(zone, idxs, hits, normals, flip):
        if len(idxs) < 3:
            return
        pts, nrm = hits[idxs], normals[idxs]
        if flip:
            pts, nrm = pts[::-1], nrm[::-1]
        strokes[zone].append(tool_poses(pts, nrm, args.standoff))

    # ---- top surfaces: hood / roof / trunk  (strokes along Y, rays go straight down)
    xs = np.arange(lo[0] + margin, hi[0] - margin + 1e-6, args.pass_spacing)
    ys = np.arange(lo[1] + margin, hi[1] - margin + 1e-6, args.step)
    top_runs = []
    for i, x in enumerate(xs):
        origins = np.array([[x, y, hi[2] + 500.0] for y in ys])
        hits, normals = cast_rays(mesh, origins, [0, 0, -1])
        valid = ~np.isnan(hits[:, 0])
        valid[valid] &= normals[valid, 2] > 0.5            # only surfaces that face upwards
        for run in split_runs(valid, hits, max_gap):
            top_runs.append((x, i, hits, normals, run))

    # The cabin (windshield + roof + rear glass) is wherever the body rises above the deck lines.
    cabin_z = args.cabin_height * hi[2]
    cabin_x = [x for (x, _, hits, _, run) in top_runs if hits[run][:, 2].max() >= cabin_z]
    hood_x = args.hood_x if args.hood_x is not None else (max(cabin_x) if cabin_x else hi[0])
    trunk_x = args.trunk_x if args.trunk_x is not None else (min(cabin_x) if cabin_x else lo[0])
    print("zone split along X: trunk < %.0f mm < roof < %.0f mm < hood" % (trunk_x, hood_x))
    for (x, i, hits, normals, run) in top_runs:
        zone = "hood" if x > hood_x else ("trunk" if x < trunk_x else "roof")
        add(zone, run, hits, normals, flip=(i % 2 == 1))

    # ---- side panels (strokes vertical, rays go sideways)
    zs = np.arange(args.sill, hi[2] - margin + 1e-6, args.step)
    for sign, zone in ((-1, "right_side"), (+1, "left_side")):
        for i, x in enumerate(xs):
            origins = np.array([[x, sign * (hi[1] + 500.0), z] for z in zs])
            hits, normals = cast_rays(mesh, origins, [0, -sign, 0])
            valid = ~np.isnan(hits[:, 0])
            valid[valid] &= (normals[valid, 1] * sign > 0.5) & (np.abs(normals[valid, 2]) < 0.5)
            for run in split_runs(valid, hits, max_gap):
                add(zone, run, hits, normals, flip=(i % 2 == 1))

    # ---- front / rear panels
    ys2 = np.arange(lo[1] + margin, hi[1] - margin + 1e-6, args.pass_spacing)
    for sign, zone in ((+1, "front"), (-1, "rear")):
        for i, y in enumerate(ys2):
            origins = np.array([[sign * (hi[0] + 500.0), y, z] for z in zs])
            hits, normals = cast_rays(mesh, origins, [-sign, 0, 0])
            valid = ~np.isnan(hits[:, 0])
            valid[valid] &= (normals[valid, 0] * sign > 0.5) & (np.abs(normals[valid, 2]) < 0.5)
            for run in split_runs(valid, hits, max_gap):
                add(zone, run, hits, normals, flip=(i % 2 == 1))
    return strokes


# ------------------------------------------------------------------ reporting
def summarise(strokes, speed):
    print("\n%-12s %8s %8s %10s" % ("zone", "strokes", "points", "length m"))
    total_len = 0.0
    for z in ZONE_ORDER:
        n_pts = sum(len(s) for s in strokes[z])
        length = 0.0
        for s in strokes[z]:
            p = np.array([T[:3, 3] for T in s])
            length += np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        total_len += length
        print("%-12s %8d %8d %10.1f" % (z, len(strokes[z]), n_pts, length / 1000))
    print("-" * 42)
    print("total painting path %.1f m  ->  about %.1f min of spraying at %d mm/s"
          % (total_len / 1000, total_len / speed / 60, speed))


def make_preview(mesh, strokes, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 9))
    views = [("Top view (X-Y)", 0, 1, 221), ("Side view (X-Z)", 0, 2, 222), ("Front view (Y-Z)", 1, 2, 223)]
    rng = np.random.default_rng(0)
    cloud = mesh.vertices[rng.choice(len(mesh.vertices), min(6000, len(mesh.vertices)), replace=False)]
    for title, a, b, pos in views:
        ax = fig.add_subplot(pos)
        ax.scatter(cloud[:, a], cloud[:, b], s=1, c="#cccccc")
        for z in ZONE_ORDER:
            for s in strokes[z]:
                p = np.array([T[:3, 3] for T in s])
                ax.plot(p[:, a], p[:, b], "-", lw=0.8, color=ZONE_COLORS[z])
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("XYZ"[a] + " (mm)")
        ax.set_ylabel("XYZ"[b] + " (mm)")
    ax3 = fig.add_subplot(224, projection="3d")
    ax3.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=1, c="#dddddd")
    for z in ZONE_ORDER:
        pts = [np.array([T[:3, 3] for T in s]) for s in strokes[z]]
        for p in pts:
            ax3.plot(p[:, 0], p[:, 1], p[:, 2], lw=0.7, color=ZONE_COLORS[z])
        if pts:
            ax3.plot([], [], color=ZONE_COLORS[z], label=z)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_title("3D (gun tip positions)")
    lo, hi = mesh.bounds
    ax3.set_xlim(lo[0] - 400, hi[0] + 400); ax3.set_ylim(lo[1] - 400, hi[1] + 400); ax3.set_zlim(0, hi[2] + 500)
    ax3.set_box_aspect((hi[0] - lo[0] + 800, hi[1] - lo[1] + 800, hi[2] + 500))
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--car", default=os.path.join(ROOT, "models", "car.stl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paths", "paths.json"))
    ap.add_argument("--standoff", type=float, default=300.0, help="gun-to-surface distance, mm")
    ap.add_argument("--pass-spacing", type=float, default=200.0, help="distance between parallel strokes, mm")
    ap.add_argument("--step", type=float, default=60.0, help="distance between waypoints along a stroke, mm")
    ap.add_argument("--margin", type=float, default=60.0, help="stay this far from panel edges, mm")
    ap.add_argument("--sill", type=float, default=250.0, help="lowest height painted on the sides/ends, mm")
    ap.add_argument("--cabin-height", type=float, default=0.68,
                    help="fraction of car height above which a surface counts as cabin (roof/glass)")
    ap.add_argument("--hood-x", type=float, default=None, help="override: top surfaces beyond this X (mm) are 'hood'")
    ap.add_argument("--trunk-x", type=float, default=None, help="override: top surfaces below this X (mm) are 'trunk'")
    ap.add_argument("--speed", type=float, default=250.0, help="painting speed used for the time estimate, mm/s")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    mesh = trimesh.load(args.car, force="mesh")
    print("car:", os.path.normpath(args.car), "bounds (mm):", np.round(mesh.bounds, 0).tolist())

    strokes = build_strokes(mesh, args)
    summarise(strokes, args.speed)

    data = {
        "meta": {
            "standoff_mm": args.standoff, "pass_spacing_mm": args.pass_spacing, "step_mm": args.step,
            "car_bounds_mm": np.round(mesh.bounds, 1).tolist(), "zone_order": ZONE_ORDER,
            "pose_format": "4x4 row-major, tool pose in the car frame (mm)",
        },
        "zones": {z: [[np.round(T.flatten(), 4).tolist() for T in s] for s in strokes[z]] for z in ZONE_ORDER},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f)
    print("\nwrote", os.path.normpath(args.out))

    if not args.no_preview:
        png = os.path.splitext(args.out)[0] + "_preview.png"
        make_preview(mesh, strokes, png)
        print("wrote", os.path.normpath(png))


if __name__ == "__main__":
    main()

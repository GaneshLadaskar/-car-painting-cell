"""
Generates the simple geometry of the painting cell as STL files (millimetres).

World frame used everywhere in this project
    origin : middle of the booth, on the floor
    X      : along the conveyor (entry on the left, exit on the right)
    Y      : across the booth (control rooms are at +Y)
    Z      : up

Layout follows the reference drawing in the assessment (12 m x 8 m cell,
8 m x 5 m painting booth, two 2 m lanes for entry/exit, three rooms at the back).

    python tools/make_cell_parts.py
"""
import json
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "models")

PED_H = 400.0     # robot pedestal height
CONV_H = 150.0    # conveyor / skid height (the car sits on top of it)
GUN_LEN = 240.0   # flange -> nozzle tip, this is where the TCP goes
GAP = 2.0         # parts that only rest on each other get a 2 mm gap so the collision checker stays quiet


def box(x0, x1, y0, y1, z0, z1):
    b = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return b


def cylinder_z(radius, z0, z1, sections=32):
    c = trimesh.creation.cylinder(radius=radius, height=z1 - z0, sections=sections)
    c.apply_translation([0, 0, (z0 + z1) / 2])
    return c


def build_gun():
    adapter = cylinder_z(60, 0, 30)
    body = cylinder_z(38, 30, 170)
    tank = box(-25, 25, 45, 105, 40, 150)               # small paint cup on the side
    cone = trimesh.creation.cone(radius=32, height=GUN_LEN - 170, sections=32)
    cone.apply_translation([0, 0, 170])                 # base at z=170, tip at z=GUN_LEN
    return trimesh.util.concatenate([adapter, body, tank, cone])


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = {
        "floor": box(-6000, 6000, -3000, 5000, -100 - GAP, -GAP),
        "conveyor": box(-6000, 6000, -700, 700, 0, CONV_H),
        # three rooms along the back wall: 3000 / 2500 / 4000 mm wide, 2500 mm deep
        "control_rooms": trimesh.util.concatenate([
            box(-4750, -1750, 2500, 5000, 0, 2500),
            box(-1750, 750, 2500, 5000, 0, 2500),
            box(750, 4750, 2500, 5000, 0, 2500),
        ]),
        "fence": box(-4000, 4000, -2750, -2700, 0, 1200),
        # pedestal is positioned relative to the robot base frame, so it hangs below Z = 0 (its bottom ends on the floor)
        "pedestal": box(-350, 350, -350, 350, -PED_H, -GAP),
        "gun": build_gun(),
    }
    for name, mesh in parts.items():
        path = os.path.join(OUT, name + ".stl")
        mesh.export(path)
        print("wrote %-14s %5d faces" % (name + ".stl", len(mesh.faces)))

    cfg = {"ped_h": PED_H, "conv_h": CONV_H, "gun_len": GUN_LEN, "gap": GAP}
    with open(os.path.join(OUT, "cell_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote cell_config.json", cfg)


if __name__ == "__main__":
    main()

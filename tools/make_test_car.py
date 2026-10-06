"""
Builds a rough sedan out of two convex pieces so the whole pipeline can be
tested before you download a real car model.

The output is deliberately "badly prepared" (metres, Y-up, off-centre) so it
also exercises prepare_car.py.

    python tools/make_test_car.py        ->  models/test_car_raw.stl
"""
import os
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "models", "test_car_raw.stl")


def hull(points):
    return trimesh.convex.convex_hull(np.array(points, dtype=float))


def build_sedan():
    # millimetres, X = length (front is +X), Z = up
    lower = hull([
        # bottom
        [-2300, -800, 0], [2300, -800, 0], [2300, 800, 0], [-2300, 800, 0],
        # belt line
        [-2400, -950, 600], [2400, -950, 600], [2400, 950, 600], [-2400, 950, 600],
        # top of hood / trunk
        [-2300, -900, 950], [2100, -900, 950], [2100, 900, 950], [-2300, 900, 950],
    ])
    cabin = hull([
        [-1400, -820, 950], [1000, -820, 950], [1000, 820, 950], [-1400, 820, 950],
        [-500, -650, 1600], [0, -650, 1600], [0, 650, 1600], [-500, 650, 1600],
    ])
    return trimesh.util.concatenate([lower, cabin])


if __name__ == "__main__":
    car = build_sedan()
    # make it awkward on purpose: metres, Y-up, shifted
    car.apply_scale(0.001)
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])  # Z-up -> Y-up
    car.apply_transform(rot)
    car.apply_translation([3.0, 0.7, -1.2])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    car.export(OUT)
    print("wrote", os.path.normpath(OUT), "extents (m):", np.round(car.extents, 3))

"""
Prepare a downloaded car model for the painting cell.

What it does
  1. loads the model and merges all parts into one mesh
  2. rotates it so  X = length,  Y = width,  Z = up
  3. scales it to the assignment size (4800 x 1900 x 1600 mm)
  4. centres it in X/Y and puts the lowest point on Z = 0
  5. reduces the triangle count (RoboDK slows down on huge meshes)
  6. writes models/car.stl in millimetres

Usage
    python tools/prepare_car.py models/test_car_raw.stl
    python tools/prepare_car.py my_download.obj --max-faces 60000
    python tools/prepare_car.py my_download.glb --height-axis y --flip-x

Accepted inputs: STL, OBJ, PLY, GLB/GLTF.  (FBX is not supported - open it in
Blender and export as STL or OBJ first.)
"""
import argparse
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "models", "car.stl")
TARGET_L, TARGET_W, TARGET_H = 4800.0, 1900.0, 1600.0   # mm, from the assessment sheet
AXES = {"x": 0, "y": 1, "z": 2}


def load_single_mesh(path):
    mesh = trimesh.load(path, force="mesh")      # scenes are merged, transforms baked in
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        sys.exit("Could not read a triangle mesh from " + path)
    mesh.merge_vertices()
    return mesh


def axis_permutation(extents, height_axis=None):
    """Return the source axis index for new X (length), Y (width), Z (height)."""
    idx = list(np.argsort(-extents))              # biggest first: length, width, height
    if height_axis is not None:
        h = AXES[height_axis]
        rest = sorted([i for i in range(3) if i != h], key=lambda i: -extents[i])
        idx = [rest[0], rest[1], h]
    return idx


def is_upside_down(mesh):
    """A car body is wider near the bottom than at the roof."""
    z = mesh.vertices[:, 2]
    lo, hi = z.min(), z.max()
    span = hi - lo
    bottom = mesh.vertices[z < lo + 0.15 * span]
    top = mesh.vertices[z > hi - 0.15 * span]
    if len(bottom) < 3 or len(top) < 3:
        return False
    width = lambda v: v[:, 1].max() - v[:, 1].min()
    return width(top) > width(bottom) * 1.05


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=DEFAULT_OUT)
    ap.add_argument("--height-axis", choices=["x", "y", "z"], help="force which source axis is 'up'")
    ap.add_argument("--flip-x", action="store_true", help="swap front and rear (rotate 180 deg about Z)")
    ap.add_argument("--flip-z", action="store_true", help="turn the model upside-down")
    ap.add_argument("--stretch", action="store_true",
                    help="scale length, width and height separately so all three match exactly")
    ap.add_argument("--max-faces", type=int, default=60000)
    args = ap.parse_args()

    mesh = load_single_mesh(args.input)
    print("loaded %d faces, raw extents %s" % (len(mesh.faces), np.round(mesh.extents, 3)))

    # 1) axes -> X length, Y width, Z up
    src = axis_permutation(mesh.extents, args.height_axis)
    P = np.zeros((3, 3))
    for new_axis, old_axis in enumerate(src):
        P[new_axis, old_axis] = 1.0
    if np.linalg.det(P) < 0:                      # avoid a mirror image
        P[1, :] *= -1
    T = np.eye(4)
    T[:3, :3] = P
    mesh.apply_transform(T)

    if args.flip_z or is_upside_down(mesh):
        if not args.flip_z:
            print("model looked upside-down, flipping it (use --flip-z yourself if this was wrong)")
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    if args.flip_x:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))

    # 2) scale to millimetres
    L, W, H = mesh.extents
    if args.stretch:
        mesh.apply_transform(np.diag([TARGET_L / L, TARGET_W / W, TARGET_H / H, 1.0]))
    else:
        mesh.apply_scale(TARGET_L / L)
    L, W, H = mesh.extents
    print("scaled size: %.0f x %.0f x %.0f mm  (target %.0f x %.0f x %.0f)" %
          (L, W, H, TARGET_L, TARGET_W, TARGET_H))
    for name, got, want in (("width", W, TARGET_W), ("height", H, TARGET_H)):
        if abs(got - want) / want > 0.10:
            print("  note: %s is more than 10%% off the target; add --stretch if you want an exact match" % name)

    # 3) centre in X/Y, wheels on Z = 0
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])

    # 4) fewer triangles
    if len(mesh.faces) > args.max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=args.max_faces)
            print("simplified to %d faces" % len(mesh.faces))
        except Exception as exc:                  # decimation is optional
            print("could not simplify (%s)" % exc)
            print("  -> run  pip install fast-simplification  and repeat, or reduce the model in Blender "
                  "(Decimate modifier). A mesh with hundreds of thousands of faces makes RoboDK very slow.")

    mesh.fix_normals()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    mesh.export(args.output)
    print("wrote", os.path.normpath(args.output))
    print("bounds (mm):", np.round(mesh.bounds, 0).tolist())


if __name__ == "__main__":
    main()

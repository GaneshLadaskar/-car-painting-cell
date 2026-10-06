# Robotic Car Painting Cell Simulation

Practical assessment for the Robotics Simulation Intern role at Yuktii AI Labs.

| | |
|---|---|
| **Name** | Ganesh Ladaskar |
| **Software used** | RoboDK `<<FILL: version, e.g. 5.x.x>>` with Python API |
| **Robot selected** | `<<FILL: exact name from the RoboDK library>>` |

> Everything marked `<<FILL: ...>>` must be replaced with your own details before you publish this repo.

## What this project does

A single industrial robot paints the exterior of a car body (about 4800 x 1900 x 1600 mm) inside a
12 m x 8 m paint cell. The cell layout follows the reference drawing from the assessment sheet:
a conveyor lane through the middle, a fenced booth and three rooms along the back wall.

The painting paths are not drawn by hand. They are generated from the car mesh:

1. `prepare_car.py` orients, scales and simplifies any downloaded car model to the target size.
2. `generate_paths.py` casts rays onto the body, builds raster strokes for seven surfaces
   (hood, roof, trunk, left side, right side, front, rear) and places the spray gun 300 mm from
   the surface, pointing along the surface normal.
3. `paint_car.py` runs inside RoboDK. The car is longer than the robot's reach, so the robot base is
   moved to eight positions around the car. At each position inverse kinematics decides which part of
   every stroke is reachable; that part is painted and the rest is carried to the next position.
4. RoboDK collision checking is active the whole time and is evaluated after every waypoint.

## Repository layout

```
tools/            runs in normal Python (outside RoboDK)
  make_test_car.py     rough test sedan, to try the pipeline before using a real model
  prepare_car.py       orient / scale / simplify a downloaded car model -> models/car.stl
  make_cell_parts.py   floor, conveyor, rooms, fence, pedestal, spray gun -> models/*.stl
  generate_paths.py    stroke generation -> paths/paths.json and paths/paths_preview.png
robodk/
  paint_car.py         runs inside RoboDK: builds the cell, plans reachability, paints, logs
simulation/           the RoboDK station (CarPaintingCell.rdk)
results/              run_log.csv, summary.json, collision_events.json written by the run
docs/screenshots/     screenshots used in the report
```

## How to reproduce

Requirements: Windows, Python 3.10+, RoboDK.

```bat
git clone https://github.com/<<FILL: your-username>>/car-painting-cell.git
cd car-painting-cell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python tools\prepare_car.py path\to\your_car_model.obj
python tools\make_cell_parts.py
python tools\generate_paths.py
```

Then in RoboDK:

1. Open `simulation/CarPaintingCell.rdk`.
2. Open `robodk/paint_car.py`, set `PROJECT_DIR` to this folder and run it from RoboDK.
   Set `ANALYZE_ONLY = True` first to see the reachability report without painting.

## Results

`<<FILL: copy your real numbers from results/summary.json>>`

| Surface | Waypoints | Reachable | Painted |
|---|---|---|---|
| hood | | | |
| roof | | | |
| trunk | | | |
| right side | | | |
| left side | | | |
| front | | | |
| rear | | | |

Collision events during the full run: `<<FILL>>`

Demo video: `<<FILL: link>>`

## Credits

Car model: `<<FILL: title, author, source link and licence of the model you used>>`

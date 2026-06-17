# VAULT: Vision-Aware Unified Layer for safe Traversal 

VAULT is a two-layer runtime safety framework that combines a learned
navigation policy (NoMaD) with a reactive look-ahead avoider (VFH\*) on top
of monocular metric depth (Depth-Anything-V2). This repository contains the
deployment code for the TurtleBot 4 platform with ROS 2 Humble.

---

## 1. Repo layout

```
VAULT/
├── deployment/
│   ├── config/                 YAML configs (vfh, robot, models, nomad, …)
│   ├── model_weights/          place NoMaD / DA2 checkpoints here
│   └── src/
│       ├── explore_vfh.py        exploration node (NoMaD + VFH*)
│       ├── Object_decetion/
│       │   └── navigation_vfh.py goal-oriented node (adds YOLO state machine)
│       ├── pd_controller.py      single-rate PD controller
│       ├── utils.py              NoMaD checkpoint loader
│       ├── VfhPlus/            core lib (vfh_star, depth_processing, …)
├── intrinsic/                  camera intrinsics per robot
├── tb4_bridge/                 ROS 2 ↔ TB4 bridge + RViz layout
├── train/vint_train/           runtime-only NoMaD model code
├── run_exploration.sh          launches exploration mode
└── run_navigation.sh           launches goal-oriented mode
```

---

## 2. Hardware & software prerequisites

- **Robot**: iRobot TurtleBot 4 with the OAK-D camera (front-facing).
- **Workstation**: Ubuntu 22.04 with ROS 2 Humble and a CUDA-capable GPU.
- **Conda env**: the launch scripts assume an env named `VAULT`. Create it
  with the dependencies listed under §4.

---

## 3. External dependencies (not vendored)

The following must be installed alongside this repo:

| Dependency | Where it goes | Why |
|---|---|---|
| [`Depth-Anything-V2`](https://github.com/DepthAnything/Depth-Anything-V2) | `VAULT/Depth-Anything-V2/` (sibling of `deployment/`) | depth backbone; `explore_vfh.py` adds its `metric_depth/` to `sys.path` automatically |
| [`diffusion_policy`](https://github.com/real-stanford/diffusion_policy) | importable on `PYTHONPATH` | NoMaD's `ConditionalUnet1D` |
| `ultralytics` (`pip install ultralytics`) | conda env | YOLOv8 for the goal-oriented node |
| Other Python deps | conda env | `torch`, `torchvision`, `numpy`, `opencv-python`, `cv-bridge`, `efficientnet-pytorch`, `diffusers`, `pyyaml`, `matplotlib`, `Pillow`, `prettytable`, `tqdm` |

### Model weights (drop into `deployment/model_weights/`)

- `nomad.pth` — pretrained NoMaD checkpoint (from the NoMaD release).
- `depth_anything_v2_metric_hypersim_vits.pth` — DA2 metric ViT-S checkpoint
  (Hypersim variant).

### YOLO weights

`yolov8n.pt` — placed at `$HOME/yolov8n.pt` by default. Override with:

```bash
YOLO_WEIGHTS=/path/to/your/yolov8n.pt ./run_navigation.sh
```

---

## 4. First-time setup

1. Clone this repo to your home directory:
   ```bash
   cd ~ && git clone <this-repo-url> VAULT
   ```
2. Clone Depth-Anything-V2 alongside:
   ```bash
   cd ~/VAULT
   git clone https://github.com/DepthAnything/Depth-Anything-V2.git
   ```
3. Create + activate the conda env (example):
   ```bash
   conda create -n CARE python=3.10
   conda activate CARE
   pip install torch torchvision diffusers efficientnet-pytorch \
               ultralytics opencv-python pyyaml matplotlib pillow \
               prettytable tqdm
   # plus your ROS 2 Humble Python bindings (rclpy, cv_bridge, sensor_msgs, …)
   ```
4. Install `diffusion_policy` so it is importable (clone next to the repo
   and add to `PYTHONPATH`, or `pip install -e .` from its source).
5. Download the model weights into `deployment/model_weights/`.
6. **Replace `<USER>` placeholders** in the configs:
   ```bash
   grep -rl "<USER>" deployment/config | xargs sed -i "s|<USER>|$USER|g"
   ```
   This rewrites:
   - `deployment/config/vfh.yaml` → `depth_weights:` path
   - `deployment/config/models.yaml` → NoMaD config + ckpt + YOLO paths
   - `deployment/src/Object_decetion/Object_detection.py` (if you want to
     hardcode it; otherwise pass `--yolo-weights`)

---

## 5. Running

Both scripts open four `gnome-terminal` panes: bridge, controller,
navigation, RViz.

### Exploration (no goal — pure NoMaD + VFH\*)

```bash
cd ~/VAULT
./run_exploration.sh
```

### Goal-oriented (NoMaD + VFH\* + YOLO target lock)

```bash
cd ~/VAULT
./run_navigation.sh
```

The default target class is `56` (chair). Change it in
`run_navigation.sh` (`--yolo-classes`) or `deployment/config/models.yaml`.

To stop everything, close the four terminal windows or `Ctrl-C` each pane.

---

## 6. RViz layout

Open `tb4_bridge/tb4_rviz.rviz` (the launch script does this for you). The
preconfigured displays cover:

| Display | Topic | What it shows |
|---|---|---|
| RobotModel + TF + Map + LaserScan | (standard) | base TB4 view |
| MarkerArray | `/vfh/depth_markers` | full polar depth fan (per-bin arrows) |
| MarkerArray | `/vfh/nomad_reference_bins` | NoMaD trajectory reference rays |
| MarkerArray | `/vfh/chosen_bin` | bin VFH\* selected |
| MarkerArray | `/vfh/goal_reference_bins` | goal-direction bins (NAV_GOAL only) |
| MarkerArray | `/vfh/detected_objects_ray` | YOLO bins (NAV_GOAL only) |
| Image | `/robot2/trajectory_viz` | camera HUD with NoMaD/VFH\* overlay |

### Save the camera HUD as a video

```bash
ros2 run image_view video_recorder --ros-args \
  -r image:=/robot2/trajectory_viz \
  -p filename:=trajectory.avi -p fps:=8.0 -p codec:=MJPG
```

### Save the RViz 3D view as a video

RViz has no built-in recorder; screen-grab the window:

```bash
WID=$(xdotool selectwindow)            # click the RViz window
ffmpeg -f x11grab -framerate 30 -window_id $WID \
       -c:v libx264 -pix_fmt yuv420p rviz_scene.mp4
```

Save a single frame instead via **File → Save Image…** in RViz.

---

## 7. Configuration reference

All tunables live under `deployment/config/`:

- **`vfh.yaml`** — VFH\* parameters: $\mu_1, \mu_2, \mu_3$, safety threshold,
  recovery cycles, depth scale, DA2 weights path.
- **`vfh_navigation.yaml`** — goal-mode VFH\* (inference/control rates,
  watchdog, executor threads).
- **`robot.yaml`** — robot radius, frame rate, `max_v`, `max_w`.
- **`camera_front.yaml` / `camera_reverse.yaml`** — camera intrinsics + framerate.
- **`models.yaml`** — NoMaD config + checkpoint paths, YOLO config.
- **`nomad.yaml`** — NoMaD model architecture (read by `utils.py:load_model`).
- **`slam.yaml`** — SLAM toolbox parameters.

---

## 8. Notes

- `run_exploration.sh` and `run_navigation.sh` activate the `CARE` conda
  env and `cd ~/VAULT`. Adjust those two lines if your repo lives
  elsewhere.
- The deployment pipeline is single-camera, single-GPU. The goal-oriented
  node uses a multithreaded executor (4 threads by default) so DA2 and
  NoMaD do not block VFH\*.
- `frame_rate` in `robot.yaml` caps the exploration loop. For stronger
  GPUs, raise it after profiling actual inference time.

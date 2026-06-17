#!/bin/bash

SETUP="source ~/miniconda3/bin/activate && conda activate CARE && export ROS_DOMAIN_ID=3 && cd ~/VfhPlus"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-$HOME/yolov8n.pt}"

if [ ! -f "$YOLO_WEIGHTS" ]; then
    echo "YOLO weights not found: $YOLO_WEIGHTS"
    echo "Set path before launch, e.g.:"
    echo "  YOLO_WEIGHTS=$HOME/yolov8n.pt  ./run_navigation.sh"
    exit 1
fi

# Terminal 1: ROS bridge
gnome-terminal --title="Bridge" -- bash -c "$SETUP && cd tb4_bridge && ./run_bridge.sh; exec bash"

# Terminal 2: Controller
gnome-terminal --title="Controller" -- bash -c "$SETUP  && python deployment/src/pd_controller.py --robot turtlebot4 --control vfh; exec bash"

# Terminal 3: Navigation
gnome-terminal --title="Navigation" -- bash -c "$SETUP  && python deployment/src/Object_decetion/navigation_vfh.py \
                          --robot turtlebot4 \
                          --yolo-weights $YOLO_WEIGHTS \
                          --yolo-conf 0.3 \
                          --yolo-classes 56 \
                          --goal-timeout-frames 4 \
                          --goal-stale-frames 2 \
                          --goal-reach-distance 0.05; exec bash"

# Terminal 4: RViz
gnome-terminal --title="RViz" -- bash -c "$SETUP && cd tb4_bridge && rviz2 -d tb4_rviz.rviz; exec bash"

#!/bin/bash

SETUP="source ~/miniconda3/bin/activate && conda activate CARE && export ROS_DOMAIN_ID=3 && cd ~/VfhPlus"

# Terminal 1: ROS bridge
gnome-terminal --title="Bridge" -- bash -c "$SETUP && cd tb4_bridge && ./run_bridge.sh; exec bash"

# Terminal 2: Controller
gnome-terminal --title="Controller" -- bash -c "$SETUP  && python deployment/src/pd_controller.py --robot turtlebot4 --control vfh; exec bash"

# Terminal 3: Navigation
gnome-terminal --title="Navigation" -- bash -c "$SETUP  && python deployment/src/explore_vfh.py --model nomad --robot turtlebot4; exec bash"

# Terminal 4: RViz
gnome-terminal --title="RViz" -- bash -c "$SETUP && cd tb4_bridge && rviz2 -d tb4_rviz.rviz; exec bash"

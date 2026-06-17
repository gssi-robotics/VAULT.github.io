#!/bin/bash

set -e

export ROS_DOMAIN_ID=3
echo "=== TurtleBot4 02493 Namespace Bridge ==="
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "DDS: $RMW_IMPLEMENTATION"
echo "=========================================="
echo ""
echo "Bridging /Turtlebot_02493/* → standard namespace"
echo "Press Ctrl+C to stop"
echo ""

ros2 service call /Turtlebot_02493/oakd/start_camera std_srvs/srv/Trigger

echo "camera triggered"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/tb4_bridge_node.py"



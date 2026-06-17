#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)
from rclpy.executors import MultiThreadedExecutor

# Message types
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image, CameraInfo, LaserScan, Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String

try:
    from irobot_create_msgs.msg import HazardDetectionVector
    HAS_IROBOT_MSGS = True
except ImportError:
    HAS_IROBOT_MSGS = False



QOS_BEST_EFFORT = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

QOS_TF_PUB = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
)

QOS_RELIABLE = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

QOS_TF_STATIC = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=100,  # buffer static transforms
)

QOS_TRANSIENT_LOCAL = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)



NS = "/Turtlebot_02493"


RELAY_TABLE = [
    # TF
    (f"{NS}/tf",           "/tf",           TFMessage,   QOS_BEST_EFFORT, QOS_TF_PUB,      "from_robot"),
    (f"{NS}/tf_static",    "/tf_static",    TFMessage,   QOS_TF_STATIC,   QOS_TF_STATIC,   "from_robot"),
    # Camera (OAK-D)
    (f"{NS}/oakd/rgb/preview/image_raw",  "/oakd/rgb/preview/image_raw",  Image,      QOS_BEST_EFFORT, QOS_RELIABLE, "from_robot"),
    (f"{NS}/oakd/rgb/preview/camera_info", "/oakd/rgb/preview/camera_info", CameraInfo, QOS_BEST_EFFORT, QOS_BEST_EFFORT, "from_robot"),
    (f"{NS}/stereo/depth", "/stereo/depth", Image,       QOS_BEST_EFFORT, QOS_RELIABLE, "from_robot"),
    # LiDAR
    (f"{NS}/scan",         "/scan",         LaserScan,   QOS_RELIABLE,    QOS_RELIABLE,    "from_robot"),
    # Odometry & IMU
    (f"{NS}/odom",         "/odom",         Odometry,    QOS_BEST_EFFORT, QOS_RELIABLE, "from_robot"),
    (f"{NS}/imu",          "/imu",          Imu,         QOS_BEST_EFFORT, QOS_BEST_EFFORT, "from_robot"),
    # Robot description (URDF for RViz)
    (f"{NS}/robot_description", "/robot_description", String, QOS_TRANSIENT_LOCAL, QOS_TRANSIENT_LOCAL, "from_robot"),
    # Joint states (RViz robot model)
    (f"{NS}/joint_states", "/joint_states", JointState,  QOS_BEST_EFFORT, QOS_BEST_EFFORT, "from_robot"),
    # Command velocity (TO the robot)
    ("/cmd_vel",           f"{NS}/cmd_vel", Twist,       QOS_RELIABLE,    QOS_RELIABLE,    "to_robot"),

    (f"{NS}/oakd/rgb/preview/image_raw", "/robot2/oakd/rgb/preview/image_raw", Image, QOS_BEST_EFFORT, QOS_RELIABLE, "from_robot"),
    ("/robot2/cmd_vel",    f"{NS}/cmd_vel", Twist,       QOS_RELIABLE,    QOS_RELIABLE,    "to_robot"),
]

if HAS_IROBOT_MSGS:
    RELAY_TABLE.append(
        (f"{NS}/hazard_detection", "/hazard_detection", HazardDetectionVector, QOS_BEST_EFFORT, QOS_BEST_EFFORT, "from_robot")
    )


class TB4BridgeNode(Node):

    def __init__(self):
        super().__init__("tb4_bridge")
        self.get_logger().info(f"Starting TurtleBot4 bridge: {NS} ↔ standard namespace")

        self._relays = []  
        self._msg_counts = {}  

        for ns_topic, std_topic, msg_type, sub_qos, pub_qos, direction in RELAY_TABLE:
            if direction == "from_robot":
                sub_topic = ns_topic
                pub_topic = std_topic
            else:  # to_robot
                sub_topic = ns_topic   
                pub_topic = std_topic  

            publisher = self.create_publisher(msg_type, pub_topic, pub_qos)
            self._msg_counts[sub_topic] = 0

            def make_callback(pub, src, dst, node_ref):
                def callback(msg):
                    pub.publish(msg)
                    node_ref._msg_counts[src] += 1
                    count = node_ref._msg_counts[src]
                    if count == 1:
                        node_ref.get_logger().info(
                            f"  ✓ FIRST msg received: {src} → {dst}"
                        )
                    elif count % 500 == 0:
                        node_ref.get_logger().info(
                            f"  heartbeat: {src} → {dst} ({count} msgs)"
                        )
                return callback

            subscription = self.create_subscription(
                msg_type,
                sub_topic,
                make_callback(publisher, sub_topic, pub_topic, self),
                sub_qos,
            )

            self._relays.append((subscription, publisher))

            arrow = "→" if direction == "from_robot" else "←"
            self.get_logger().info(
                f"  relay: {ns_topic} {arrow} {std_topic}  "
                f"[{msg_type.__name__}, "
                f"{'BE' if sub_qos.reliability == ReliabilityPolicy.BEST_EFFORT else 'REL'}]"
            )

        self.get_logger().info(f"Bridge active with {len(self._relays)} relays")


def main(args=None):
    rclpy.init(args=args)
    node = TB4BridgeNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

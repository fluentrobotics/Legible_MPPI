import math

import rclpy.time
from geometry_msgs.msg import Transform, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class TF2Wrapper:
    def __init__(self, node: Node) -> None:
        self._node = node
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self._node)

        self._tf2_pub = self._node.create_publisher(TFMessage, "/tf", 1)

    def get_latest_pose(self, target_frame: str, source_frame: str) -> Transform | None:
        """
        Returns the transform T_target_source (interpreted using left-handed
        matrix multiplication).
        """
        try:
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),  # type: ignore
            ).transform
        except:  # noqa: E722
            return None

    def publish_2d_pose(
        self, target_frame: str, source_frame: str, x: float, y: float, theta: float
    ) -> None:
        transform = TransformStamped()
        transform.header.frame_id = target_frame
        transform.header.stamp = self._node.get_clock().now().to_msg()
        transform.child_frame_id = source_frame

        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.rotation.z = math.sin(theta / 2)
        transform.transform.rotation.w = math.cos(theta / 2)

        self._tf2_pub.publish(TFMessage(transforms=[transform]))

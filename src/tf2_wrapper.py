from geometry_msgs.msg import Transform
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class TF2Wrapper:
    def __init__(self, node: Node) -> None:
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)

    def get_latest_pose(self, target_frame: str, source_frame: str) -> Transform | None:
        try:
            return self._tf_buffer.lookup_transform(
                target_frame, source_frame, Time()
            ).transform
        except TransformException:
            return None

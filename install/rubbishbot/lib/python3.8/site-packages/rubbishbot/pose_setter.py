import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import math
import time

class PoseSetter(Node):
    def __init__(self):
        print("✅ This is the REAL pose_setter being executed.")

        super().__init__('pose_setter')

        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        # ✅ Fixed initial pose from amcl_pose
        self.x = -0.540629506111145
        self.y = -0.2792171835899353

        z = 0.8103257103648465
        w = 0.5859797292754305
        
        yaw_rad = math.atan2(2.0 * (w * z), 1.0 - 2.0 * (z * z))
        yaw_deg = math.degrees(yaw_rad)
        self.yaw_deg = yaw_deg

        self.delay_sec = 2.5
        self.timer = self.create_timer(0.1, self.publish_once)
        self.published = False

    def publish_once(self):
        if self.published:
            return

        yaw = math.radians(self.yaw_deg)
        time.sleep(self.delay_sec)

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        msg.pose.covariance[0] = 0.25    # x
        msg.pose.covariance[7] = 0.25    # y
        msg.pose.covariance[35] = 0.0685 # yaw (~15 deg)

        self.publisher.publish(msg)
        self.get_logger().info(f"✅ Initial pose published at ({self.x:.2f}, {self.y:.2f}, {yaw:.2f} rad)")
        self.published = True

def main(args=None):
    rclpy.init(args=args)
    node = PoseSetter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
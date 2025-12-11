import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Twist, PoseStamped
import math
import subprocess
import shlex
import os
import threading
import time


BASE_POSE = (-0.624, -0.0022, -0.00534)
# WAYPOINTS = [(-3.34,6.05,-0.001)]
WAYPOINTS = [(-1.71, 1.23, 0.00)]



SCAN_DEGREES = 270
SCAN_SPEED = 0.5

class State:
    IDLE = 'idle'
    NAVIGATING = 'navigating'
    SCANNING = 'scanning'
    RETURNING = 'returning'
    NAVIGATING_TO_OBJECT = 'navigating_to_object'
    DONE = 'done'

class ExplorationNode(Node):
    def __init__(self):
        super().__init__('exploration_node')
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.state = State.IDLE
        self.current_wp = 0
        self.timer = self.create_timer(1.0, self.main_loop)

        self.scan_timer = None
        self.scan_start_time = None
        self.scan_duration = None
        self.scan_twist = Twist()

        # Interrupt state
        self.object_detected = False
        self.object_pose = None
        self.interrupted = False  # To stop main exploration after object
        # self.target_pose_sub = self.create_subscription( # Published by Object detection
        #     PoseStamped, '/target_pose', self.object_detected_callback, 10)

        # self.say("Exploration initialized.")
        self.get_logger().info("🟢 Exploration node ready.")
    
    def main_loop(self):
        # PAUSE: If object detected, do NOT allow new waypoint navigation or scanning
        if self.object_detected and self.state in [State.IDLE, State.NAVIGATING, State.SCANNING]:
            # Only print this once per loop cycle
            if not self.interrupted:
                self.get_logger().info("🛑 Exploration paused due to object detection.")
                self.interrupted = True
            return

        if self.state == State.IDLE:
            if not self.object_detected and self.current_wp < len(WAYPOINTS):
                self.navigate_to(WAYPOINTS[self.current_wp])
                self.state = State.NAVIGATING
            elif not self.object_detected and self.current_wp >= len(WAYPOINTS):
                self.get_logger().info("Waypoints done.")
            # else: If object_detected, block.

        elif self.state == State.NAVIGATING:
            pass  # Waiting for navigation result

        elif self.state == State.SCANNING:
            pass  # Scan handled by scan_timer

        elif self.state == State.NAVIGATING_TO_OBJECT:
            pass  # Waiting for nav result

        elif self.state == State.RETURNING:
            pass  # Wait for nav complete

        elif self.state == State.DONE:
            # self.say("Mission complete.")
            self.get_logger().info("🏁 Mission done.")
            self.timer.cancel()
    
    def object_detected_callback(self, msg):
        if not self.object_detected:
            self.get_logger().info("🛑 Object detected! Interrupting exploration.")
            self.object_detected = True
            self.object_pose = msg
            self.cancel_current_nav_goal()
            self.stop_scanning()
            threading.Thread(target=self.navigate_to_object_after_delay, daemon=True).start()
    
    def navigate_to_object_after_delay(self):
        time.sleep(3)
        # self.say("Navigating to object.")
        self.get_logger().info("🚀 Navigating to detected object.")
        self.navigate_to_pose_msg(self.object_pose)
        self.state = State.NAVIGATING_TO_OBJECT
    
    def cancel_current_nav_goal(self):
        try:
            goal_handle = getattr(self.nav_client, '_goal_handle', None)
            if goal_handle:
                self.get_logger().info("🛑 Cancelling current navigation goal...")
        except Exception as e:
            self.get_logger().warn(f"Could not cancel nav goal: {e}")
    
    def stop_scanning(self):
        if self.scan_timer:
            self.scan_timer.cancel()
            self.scan_timer = None
            self.cmd_vel_pub.publish(Twist())  # stop rotation
            self.get_logger().info("🛑 Scan interrupted and stopped.")
    
    def navigate_to_pose_msg(self, pose_msg):
        goal = NavigateToPose.Goal()
        goal.pose = pose_msg
        def send_goal():
            future = self.nav_client.send_goal_async(goal)
            future.add_done_callback(self.handle_goal_response)
        if self.nav_client.wait_for_server(timeout_sec=2.0):
            send_goal()
        else:
            self.get_logger().warn("❌ Nav2 server not ready for object!")
    
    def navigate_to(self, target):
        # Only block navigation to waypoints, NOT base return after pick
        # Allow if returning, else block if interrupted
        if self.object_detected and self.state not in [State.RETURNING]:
            return
        x, y, yaw = target
        self.get_logger().info(f"🚀 Navigating to: ({x:.2f}, {y:.2f}, {yaw:.1f}°)")
        # self.say("Navigating")

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        yaw_rad = math.radians(yaw)
        goal.pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw_rad / 2.0)

        def send_goal():
            future = self.nav_client.send_goal_async(goal)
            future.add_done_callback(self.handle_goal_response)

        if self.nav_client.wait_for_server(timeout_sec=2.0):
            send_goal()
        else:
            self.get_logger().warn("❌ Nav2 server not ready!")
    
    def handle_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("❌ Goal rejected.")
            self.state = State.IDLE
            return

        self.get_logger().info("✅ Goal accepted.")
        self.nav_client._goal_handle = goal_handle  # Store handle for cancel
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_goal_complete)
    
    def start_rotation(self, speed=SCAN_SPEED, angle_deg=SCAN_DEGREES):
        if self.object_detected:
            return  # Do not scan if object detected!
        self.get_logger().info(f"🔄 Starting scan: {angle_deg}° at {speed} rad/s")
        self.scan_duration = math.radians(angle_deg) / speed
        self.scan_start_time = self.get_clock().now().nanoseconds / 1e9
        twist = Twist()
        twist.angular.z = speed
        self.scan_twist = twist
        self.scan_timer = self.create_timer(0.1, self.scan_loop)
        self.state = State.SCANNING
    
    def scan_loop(self):
        if self.object_detected:
            self.cmd_vel_pub.publish(Twist())  # stop
            if self.scan_timer:
                self.scan_timer.cancel()
                self.scan_timer = None
            self.get_logger().info("🛑 Scan interrupted.")
            return
        now = self.get_clock().now().nanoseconds / 1e9
        elapsed = now - self.scan_start_time
        if elapsed < self.scan_duration:
            self.cmd_vel_pub.publish(self.scan_twist)
        else:
            self.cmd_vel_pub.publish(Twist())  # stop
            if self.scan_timer:
                self.scan_timer.cancel()
                self.scan_timer = None
            self.get_logger().info("✅ Scan complete.")
            # self.say("Scan complete")
            self.current_wp += 1
            self.state = State.IDLE

    def on_goal_complete(self, future):
        result = future.result().result
        status = future.result().status
        
        # Status 4 means SUCCEEDED
        if status == 4:
            self.get_logger().info('✅ Navigation succeeded!')
            
            # Logic to decide what to do next
            if self.state == State.NAVIGATING:
                self.start_rotation() # Start scanning
            elif self.state == State.NAVIGATING_TO_OBJECT:
                self.get_logger().info("Arrived at object. Handling interaction...")
                # Add object handling logic here
            elif self.state == State.RETURNING:
                self.state = State.DONE
        else:
            self.get_logger().warn(f'❌ Navigation failed with status: {status}')
            self.state = State.IDLE

def main(args=None):
    rclpy.init(args=args)
    node = ExplorationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
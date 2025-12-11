#!/home/agilex/ros2_rubbishbot_ws/venv_with_tensorrt/bin/python3
import math
from ultralytics import YOLO
import tf2_geometry_msgs
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped, Quaternion, PoseWithCovarianceStamped
import numpy as np
# Restore the deprecated alias so cv_bridge doesn't crash
if not hasattr(np, 'bool'):
    np.bool = np.bool_
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros
import cv2

def yaw_from_quat(q):
    return math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

def quat_from_yaw(yaw):
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q

class TrashDetector(Node):
    def __init__(self):
        super().__init__('trash_detector')
        
        self.MIN_DEPTH_METERS = 0.15
        self.MAX_DEPTH_METERS = 4.0

        self.bridge = CvBridge()
        self.model = YOLO("/home/agilex/ros2_rubbishbot_ws/src/Trashbot/yolov8n4.engine") 
        self.COLOR_IMAGE_TOPIC = '/camera/color/image_raw'
        self.DEPTH_IMAGE_TOPIC = '/camera/depth/image_raw'
        self.DEPTH_CAMERA_INFO = '/camera/depth/camera_info'

        # Subscribes to /image_raw.
        self.rgb_sub = Subscriber(self, Image, self.COLOR_IMAGE_TOPIC)
        
        self.depth_sub = Subscriber(self, Image, self.DEPTH_IMAGE_TOPIC)
        self.info_sub = Subscriber(self, CameraInfo, self.DEPTH_CAMERA_INFO)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_pose_callback, 10)
        
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub],
            queue_size=10,
            slop=0.1
        )

        self.ts.registerCallback(self.image_callback)

        # Publishes an annotated image with bounding boxes
        self.visualization_pub = self.create_publisher(Image, '/rubbishbot/debug_image', 10)
        self.target_pub = self.create_publisher(PoseStamped, '/rubbishbot/target_pose', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info('Trash Detector Node has been started.')

    def image_callback(self, rgb_sub, depth_sub, info_sub):
        try:
            # Run detection
            detections = self.detect_trash(rgb_sub, depth_sub, info_sub)
            if detections is None:
                detections = []
            else:
                cv_image = self.bridge.imgmsg_to_cv2(rgb_sub, "bgr8")
                # Draw detections on the image for visualization
                annotated_frame = self.draw_detections(cv_image, detections)
                
                # Publish the annotated image
                out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
                self.visualization_pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def amcl_pose_callback(self, msg):
        self.latest_amcl_pose = msg.pose.pose

    def get_3d_point(self, depth_img, cx, cy):
        points = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = cx + dx, cy + dy
                if 0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]:
                    z = depth_img[y, x] / 1000.0  # mm to m
                    if self.MIN_DEPTH_METERS < z < self.MAX_DEPTH_METERS:
                        X = (x - self.cx) * z / self.fx
                        Y = (y - self.cy) * z / self.fy
                        points.append((X, Y, z))
        if not points:
            return None
        return tuple(np.median(points, axis=0))

    def transform_to_map(self, point, stamp, src_frame):
        p = PointStamped()
        p.header.stamp = stamp
        p.header.frame_id = src_frame
        p.point.x, p.point.y, p.point.z = point

        try:
            return self.tf_buffer.transform(p, "map", timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception as e1:
            self.get_logger().warn(f"[⚠️ TF TIME WARN] {e1} — falling back to 'now'")
            # Try again with current time (safe fallback)
            p.header.stamp = rclpy.time.Time().to_msg()
            try:
                return self.tf_buffer.transform(p, "map", timeout=rclpy.duration.Duration(seconds=0.5))
            except Exception as e2:
                self.get_logger().error(f"[❌ TF FAIL] Even 'now' failed: {e2}")
                return None

    def detect_trash(self, rgb_image_msg, depth_image_msg, cam_info_msg):
        self.fx, self.fy = cam_info_msg.k[0], cam_info_msg.k[4]
        self.cx, self.cy = cam_info_msg.k[2], cam_info_msg.k[5]

        if not self.latest_amcl_pose:
            return []
        
        color_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "16UC1")

        results = self.model(color_image, imgsz=640, conf=0.7, verbose=False)

        if not results or not results[0].boxes:
            return []
        
        # Sort boxes by height (largest vertical trash item)
        boxes = sorted(results[0].boxes, key=lambda b: b.xyxy[0][3]-b.xyxy[0][1], reverse=True)
        best_box = boxes[0]
        
        # Extract data
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        cls_id = int(best_box.cls[0])
        conf = float(best_box.conf[0])
        cls_name = self.model.names[cls_id]
        
        # --- SAFETY CLAMP: Ensure coordinates are inside image ---
        h, w = depth_image.shape
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        
        self.get_logger().info(f"[DEBUG] Pixel ({cx},{cy}) depth={depth_image[cy, cx]}mm")

        point_3d = self.get_3d_point(depth_image, cx, cy)
        if not point_3d:
            self.get_logger().warn(f"[SKIP] No valid depth at ({cx},{cy})")
            return []

        x_cam, y_cam, z_cam = point_3d
        
        src_frame = depth_image_msg.header.frame_id.strip()
        tf_point = self.transform_to_map(point_3d, depth_image_msg.header.stamp, src_frame)
        if not tf_point:
            return []

        # --- NAVIGATION LOGIC ---
        CAMERA_TO_BASE_LINK = 0.12 
        amcl_x = self.latest_amcl_pose.position.x
        amcl_y = self.latest_amcl_pose.position.y
        obj_x, obj_y = tf_point.point.x, tf_point.point.y
        
        dx = obj_x - amcl_x
        dy = obj_y - amcl_y
        d = math.hypot(dx, dy)
        d = max(d, 1e-3)
        
        approach_x = obj_x - (dx / d) * CAMERA_TO_BASE_LINK
        approach_y = obj_y - (dy / d) * CAMERA_TO_BASE_LINK
        
        goal_yaw = math.atan2(obj_y - approach_y, obj_x - approach_x)
        q = quat_from_yaw(goal_yaw)

        # Publish navigation goal
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = rgb_image_msg.header.stamp
        pose_msg.pose.position.x = approach_x
        pose_msg.pose.position.y = approach_y
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation = q
        
        self.target_pub.publish(pose_msg)
        self.get_logger().info(f"[📍 DETECTED] {cls_name} 2D({cx},{cy})")

        # --- CRITICAL FIX: Return a LIST OF DICTIONARIES ---
        # This matches what your draw_detections function expects
        detection_list = [{
            'bbox': [x1, y1, x2, y2],
            'confidence': conf,
            'class_id': cls_id
        }]

        return detection_list

    def draw_detections(self, img, detections):
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw Label
            label = f"Trash: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

def main(args=None):
    rclpy.init(args=args)
    node = TrashDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
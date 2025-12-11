#!/home/agilex/ros2_rubbishbot_ws/venv_with_tensorrt/bin/python3
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
# Restore the deprecated alias so cv_bridge doesn't crash
if not hasattr(np, 'bool'):
    np.bool = np.bool_
from cv_bridge import CvBridge
import cv2

class TrashDetector(Node):
    def __init__(self):
        super().__init__('trash_detector')

        self.bridge = CvBridge()
        self.model = YOLO("/home/agilex/ros2_rubbishbot_ws/src/Trashbot/yolov8n4.engine") 
        self.COLOR_IMAGE_TOPIC = '/camera/color/image_raw'
        # Subscribes to /image_raw.
        self.subscription = self.create_subscription(
            Image,
            self.COLOR_IMAGE_TOPIC,
            self.image_callback,
            10)

        # Publishes an annotated image with bounding boxes
        self.publisher_ = self.create_publisher(Image, '/trash/debug_image', 10)

        self.get_logger().info('Trash Detector Node has been started.')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run detection
            detections = self.detect_trash(cv_image)
            
            # Draw detections on the image for visualization
            annotated_frame = self.draw_detections(cv_image, detections)
            
            # Publish the annotated image
            out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            self.publisher_.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def detect_trash(self, cv_image):
        results = self.model(cv_image, imgsz=640, conf=0.5, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf >= 0.5: 
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls_id
                    })
        return detections

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
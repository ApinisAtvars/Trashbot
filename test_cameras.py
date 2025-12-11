import cv2

def test_cameras():
    # Try indices 0 through 5
    for index in range(6):
        print(f"--- Testing /dev/video{index} ---")
        # Force V4L2 backend to avoid Obsensor errors
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        
        if cap.isOpened():
            print(f"[SUCCESS] Camera found at index {index}")
            
            # Read a frame to ensure it's actually working
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"   Resolution: {w}x{h}")
                print(f"   > USE CAMERA_SOURCE = {index}")
            else:
                print("   [WARNING] Opened, but returned empty frame (might be IR/Depth node).")
            
            cap.release()
        else:
            print(f"[FAILED] Could not open video{index}")

if __name__ == "__main__":
    test_cameras()
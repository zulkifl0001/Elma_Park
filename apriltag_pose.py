"""
AprilTag Pose Estimation
Tag Size: 16.1 cm (0.161 m)
Focus: Locked to Infinity
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import json
import os
import sys

# --- USER SETTINGS ---
TAG_SIZE = 0.161  # 16.1 cm in meters
# ---------------------

CALIBRATION_FILE = "camera_calibration.json"

def get_tag_points():
    # Define the 3D coordinates of the tag corners (centered at 0,0,0)
    half = TAG_SIZE / 2.0
    return np.array([
        [-half, -half, 0],  # Bottom-Left
        [ half, -half, 0],  # Bottom-Right
        [ half,  half, 0],  # Top-Right
        [-half,  half, 0]   # Top-Left
    ], dtype=np.float32)

def main():
    print("="*60)
    print(f"AprilTag Pose Estimation")
    print(f"Tag Size: {TAG_SIZE*100:.1f} cm")
    print("="*60)

    # 1. Load Calibration
    if not os.path.exists(CALIBRATION_FILE):
        print("Error: Calibration file not found. Run camera_calibration.py first.")
        return
    with open(CALIBRATION_FILE, 'r') as f:
        data = json.load(f)
    mtx = np.array(data["camera_matrix"])
    dist = np.array(data["distortion_coefficients"])

    # 2. Setup Detector
    try:
        import pyapriltags
        detector = pyapriltags.Detector(families="tag36h11")
        use_cv_detector = False
        print("Using pyapriltags detector.")
    except ImportError:
        detector = cv2.aruco.AprilTagDetector()
        use_cv_detector = True
        print("Using OpenCV detector.")

    # 3. Setup Camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1920, 1080), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    
    # --- LOCK FOCUS ---
    print("Locking Focus to Infinity (0.0)...")
    picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})

    obj_points = get_tag_points()
    # Use SQPNP solver if available (better accuracy)
    pnp_flags = cv2.SOLVEPNP_SQPNP if hasattr(cv2, 'SOLVEPNP_SQPNP') else cv2.SOLVEPNP_ITERATIVE

    try:
        while True:
            frame = picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            detected = False
            corners = None
            
            # Detect Tags
            if not use_cv_detector:
                results = detector.detect(gray)
                for r in results:
                    if r.tag_id == 0:
                        corners = r.corners.reshape(4, 1, 2)
                        detected = True
            else:
                det_corners, ids, _ = detector.detect(gray)
                if ids is not None:
                    for i, id in enumerate(ids):
                        if id == 0:
                            corners = det_corners[i]
                            detected = True

            # Calculate Pose
            if detected:
                success, rvec, tvec = cv2.solvePnP(obj_points, corners, mtx, dist, flags=pnp_flags)
                
                if success:
                    # Draw visual markers
                    cv2.polylines(display, [corners.astype(int)], True, (0, 255, 0), 3)
                    cv2.drawFrameAxes(display, mtx, dist, rvec, tvec, TAG_SIZE/2)
                    
                    # Display distance
                    x = tvec[0][0] * 100
                    z = tvec[2][0] * 100
                    cv2.putText(display, f"X: {x:.1f} cm", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display, f"Z: {z:.1f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Pose Estimation", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
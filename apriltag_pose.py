"""
Camera Calibration (A3 Setup)
Pattern: 9x6 Inner Corners
Square Size: 2.8 cm (0.028 m)
Focus: Locked to Infinity
"""

import cv2
import numpy as np
import os
from pathlib import Path
import json
from picamera2 import Picamera2

# --- USER SETTINGS ---
CHECKERBOARD_SIZE = (9, 6)  # Inner corners
SQUARE_SIZE = 0.028         # 2.8 cm in meters
# ---------------------

MIN_IMAGES = 15
CALIBRATION_DIR = "calibration_images"
CALIBRATION_FILE = "camera_calibration.json"

def prepare_object_points():
    objp = np.zeros((CHECKERBOARD_SIZE[1] * CHECKERBOARD_SIZE[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp

def detect_checkerboard(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    vis = image.copy()
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(vis, CHECKERBOARD_SIZE, corners, ret)
        return True, corners, vis
    return False, None, vis

def main():
    print("="*60)
    print(f"Calibration for A3 Board ({SQUARE_SIZE*100} cm squares)")
    print("="*60)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1920, 1080), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    
    # --- LOCK FOCUS ---
    print("Locking Focus to Infinity (0.0)...")
    picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})
    
    Path(CALIBRATION_DIR).mkdir(exist_ok=True)
    saved_count = 0
    
    # 1. Capture Loop
    try:
        while True:
            frame = picam2.capture_array()
            ret, corners, display = detect_checkerboard(frame)
            display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            
            status = f"Saved: {saved_count} | Press 'c' to save, 'q' to finish"
            color = (0, 255, 0) if ret else (0, 0, 255)
            cv2.putText(display, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.imshow("Calibration Capture", display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and ret:
                fname = os.path.join(CALIBRATION_DIR, f"calib_{saved_count:03d}.jpg")
                cv2.imwrite(fname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                saved_count += 1
                print(f"Saved {fname}")
            elif key == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

    if saved_count < MIN_IMAGES:
        print("Not enough images saved.")
        return

    # 2. Calibration Calculation
    print("\nCalculating Calibration...")
    objp = prepare_object_points()
    objpoints, imgpoints = [], []
    img_size = (1920, 1080)
    
    for fname in sorted(Path(CALIBRATION_DIR).glob("*.jpg")):
        img = cv2.imread(str(fname))
        ret, corners, _ = detect_checkerboard(img)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    
    if ret:
        data = {"camera_matrix": mtx.tolist(), "distortion_coefficients": dist.tolist()}
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\nSUCCESS! Calibration saved to {CALIBRATION_FILE}")
    else:
        print("Calibration failed.")

if __name__ == "__main__":
    main()

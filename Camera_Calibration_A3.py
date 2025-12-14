import cv2
import numpy as np
import os
from pathlib import Path
import json
from picamera2 import Picamera2

# --- A3 SPECS ---
CHECKERBOARD_SIZE = (9, 6)
SQUARE_SIZE = 0.028  
# ----------------

MIN_IMAGES = 15
CALIBRATION_IMAGES_DIR = "calibration_images_A3"
CALIBRATION_FILE = "camera_calibration_A3.json"

def prepare_object_points():
    objp = np.zeros((CHECKERBOARD_SIZE[1] * CHECKERBOARD_SIZE[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp

def detect_checkerboard(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    image_with_corners = image.copy()
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(image_with_corners, CHECKERBOARD_SIZE, corners_refined, ret)
        return True, corners_refined, image_with_corners
    return False, None, image_with_corners

def capture_and_calibrate():
    print("Initializing A3 Calibration (Square Size: 2.8cm)...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1920, 1080), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    
    # LOCK FOCUS
    print("Locking Focus to Infinity...")
    picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})
    
    Path(CALIBRATION_IMAGES_DIR).mkdir(exist_ok=True)
    saved_count = 0
    
    try:
        while True:
            frame = picam2.capture_array()
            ret, corners, display_frame = detect_checkerboard(frame)
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            
            msg = f"Saved: {saved_count} (Press 'c')" if ret else f"Saved: {saved_count}"
            color = (0, 255, 0) if ret else (0, 0, 255)
            cv2.putText(display_frame, msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("A3 Calibration", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and ret:
                fname = os.path.join(CALIBRATION_IMAGES_DIR, f"calib_{saved_count:03d}.jpg")
                cv2.imwrite(fname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                saved_count += 1
                print(f"Saved {fname}")
            elif key == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

    if saved_count < MIN_IMAGES:
        print("Not enough images to calibrate.")
        return

    print("Calibrating...")
    objp = prepare_object_points()
    objpoints = []
    imgpoints = []
    img_size = (1920, 1080) # Assuming full HD
    
    for fname in sorted(Path(CALIBRATION_IMAGES_DIR).glob("*.jpg")):
        img = cv2.imread(str(fname))
        ret, corners, _ = detect_checkerboard(img)
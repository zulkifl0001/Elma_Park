#!/usr/bin/env python3
import cv2
import numpy as np
import math
import time

# --- 1. CONFIGURATION ---
TAG_SIZE = 0.05  # Measure your tag! (meters)
HEADING_OFFSET = 0 # Change to 180 if car thinks North is South

# Raspberry Pi Optimization
# Set to True if you are running via SSH without a monitor attached
HEADLESS_MODE = False 

# The Map (Global Positions)
TAG_LIBRARY = {
    0: {'x': 1.0, 'y': 2.0, 'angle': 270}, # North Wall
    1: {'x': 0.0, 'y': 1.0, 'angle': 0},   # West Wall
    2: {'x': 2.0, 'y': 1.0, 'angle': 180}  # East Wall
}

# Precision Grid (10cm)
GRID_WIDTH = 2.0
GRID_HEIGHT = 2.0
CELL_SIZE = 0.1
GRID_COLS = int(GRID_WIDTH / CELL_SIZE)
GRID_ROWS = int(GRID_HEIGHT / CELL_SIZE)

def get_grid_coords(x, y):
    if x is None or x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
        return None
    col = int(x / CELL_SIZE)
    row = int(y / CELL_SIZE)
    return (row, col)

def calculate_global_pose(tag_id, tvec, rvec):
    if tag_id not in TAG_LIBRARY:
        return None, None, None

    tag_config = TAG_LIBRARY[tag_id]
    
    # 1. Relative Distances
    dist_z = tvec[2][0]
    dist_x = tvec[0][0]
    flat_dist = math.sqrt(dist_z**2 + dist_x**2)
    bearing = math.atan2(dist_x, dist_z)

    # 2. Relative Orientation
    R, _ = cv2.Rodrigues(rvec)
    yaw_rel_rad = math.atan2(R[0, 2], R[2, 2]) 
    yaw_rel_deg = np.degrees(yaw_rel_rad)

    # 3. Global Heading
    car_heading = (tag_config['angle'] - yaw_rel_deg + 180 + HEADING_OFFSET) % 360

    # 4. Global Position
    theta_rad = np.radians(car_heading)
    vector_angle = theta_rad - bearing + math.pi
    
    gx = tag_config['x'] + (flat_dist * math.cos(vector_angle))
    gy = tag_config['y'] + (flat_dist * math.sin(vector_angle))

    return gx, gy, car_heading

def draw_minimap(frame, grid_coords, car_heading, active_tag_id):
    if HEADLESS_MODE: return
    
    h, w, _ = frame.shape
    map_size = 200 # Smaller map for Pi Performance
    margin = 10
    
    map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255
    cell_px = map_size / GRID_COLS

    # Draw Grid
    for i in range(GRID_COLS + 1):
        x = int(i * cell_px)
        cv2.line(map_img, (x, 0), (x, map_size), (220, 220, 220), 1)
    for i in range(GRID_ROWS + 1):
        y = int(i * cell_px)
        cv2.line(map_img, (0, y), (map_size, y), (220, 220, 220), 1)

    # Draw Tags
    for tid, config in TAG_LIBRARY.items():
        tx_px = int((config['x'] / GRID_WIDTH) * map_size)
        ty_px = int(((GRID_HEIGHT - config['y']) / GRID_HEIGHT) * map_size)
        color = (0,255,0) if tid == active_tag_id else (0,0,0)
        cv2.rectangle(map_img, (tx_px-3, ty_px-3), (tx_px+3, ty_px+3), color, -1)

    # Draw Car
    if grid_coords:
        r, c = grid_coords
        draw_row = (GRID_ROWS - 1) - r
        cx = int((c + 0.5) * cell_px)
        cy = int((draw_row + 0.5) * cell_px)
        cv2.circle(map_img, (cx, cy), 5, (0, 0, 255), -1)
        
        angle_rad = np.radians(car_heading)
        ex = int(cx + 10 * math.cos(angle_rad))
        ey = int(cy - 10 * math.sin(angle_rad))
        cv2.line(map_img, (cx, cy), (ex, ey), (255, 0, 0), 2)

    # Overlay
    y_off = h - map_size - margin
    x_off = w - map_size - margin
    frame[y_off:y_off+map_size, x_off:x_off+map_size] = map_img
    cv2.rectangle(frame, (x_off, y_off), (x_off+map_size, y_off+map_size), (0,0,0), 2)

def run_pi_simulation():
    # Use index 0. If using a USB cam and it fails, try -1 or 1.
    cap = cv2.VideoCapture(0)
    
    # PI OPTIMIZATION: Set Low Resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Check if camera opened
    if not cap.isOpened():
        print("Error: Camera not accessible. Check connection.")
        return

    # Warmup
    time.sleep(1.0)
    ret, frame = cap.read()
    if not ret: 
        print("Error: Could not read frame.")
        return
        
    h, w = frame.shape[:2]
    focal_length = w * 0.8  
    cam_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1))
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    
    obj_points = np.array([[-TAG_SIZE/2, TAG_SIZE/2, 0], [TAG_SIZE/2, TAG_SIZE/2, 0], 
                           [TAG_SIZE/2, -TAG_SIZE/2, 0], [-TAG_SIZE/2, -TAG_SIZE/2, 0]], dtype=np.float32)

    print("Pi Localization Active. Press 'q' to quit.")
    
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Performance Monitoring
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        grid_pos = None
        heading = 0
        active_id = -1
        
        if ids is not None:
            if not HEADLESS_MODE:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Use First Tag Found (Faster than sorting on Pi)
            # Or use logic to find closest if needed
            best_idx = 0 
            
            if ids[best_idx][0] in TAG_LIBRARY:
                active_id = ids[best_idx][0]
                success, rvec, tvec = cv2.solvePnP(obj_points, corners[best_idx][0], cam_matrix, dist_coeffs)
                
                if success:
                    gx, gy, gth = calculate_global_pose(active_id, tvec, rvec)
                    
                    if gx is not None:
                        grid_pos = get_grid_coords(gx, gy)
                        heading = gth
                        
                        # Console Output (Useful for Pi debugging via SSH)
                        print(f"Tag:{active_id} | Pos:({gx:.2f}, {gy:.2f}) | Grid:{grid_pos}")
                        
                        if not HEADLESS_MODE:
                            color = (0, 255, 0) if grid_pos else (0, 0, 255)
                            cv2.putText(frame, f"Pos: ({gx:.2f}, {gy:.2f})", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            cv2.putText(frame, f"FPS: {int(fps)}", (10, 470), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, 0.1)

        if not HEADLESS_MODE:
            draw_minimap(frame, grid_pos, heading, active_id)
            cv2.imshow("Pi Parking", frame)
            if cv2.waitKey(1) == ord('q'): break
        else:
            # If headless, add a small sleep to prevent 100% CPU usage loop
            time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pi_simulation()

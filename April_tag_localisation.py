import cv2
import numpy as np
import math
import time

# --- 1. CONFIGURATION ---
TAG_SIZE = 0.05  # 5 cm (Adjust if your distance is wrong!)

# The Map of the Arena (2m x 2m)
# We define the fixed position of each tag on the walls.
TAG_LIBRARY = {
    # ID 0: North Wall (Top), Facing South
    0: {'x': 1.0, 'y': 2.0, 'wall': 'NORTH'},
    
    # ID 1: West Wall (Left), Facing East
    1: {'x': 0.0, 'y': 1.0, 'wall': 'WEST'},
    
    # ID 2: East Wall (Right), Facing West
    2: {'x': 2.0, 'y': 1.0, 'wall': 'EAST'}
}

# Grid Settings
GRID_WIDTH = 2.0
GRID_HEIGHT = 2.0
CELL_SIZE = 0.1  # 10cm cells
GRID_COLS = int(GRID_WIDTH / CELL_SIZE)
GRID_ROWS = int(GRID_HEIGHT / CELL_SIZE)

def get_grid_coords(x, y):
    """Converts meters to grid row/col."""
    if x is None: return None
    if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
        return None
    # Row is Y (0 at bottom), Col is X (0 at left)
    col = int(x / CELL_SIZE)
    row = int(y / CELL_SIZE)
    return (row, col)

def calculate_global_pose(tag_id, tvec):
    """
    Calculates Global X/Y based on which wall the tag is on.
    We assume the camera is facing the tag (approx 90 deg).
    """
    if tag_id not in TAG_LIBRARY:
        return None, None

    tag_config = TAG_LIBRARY[tag_id]
    wall = tag_config['wall']
    tx = tag_config['x']
    ty = tag_config['y']

    # Raw Camera Data
    z_raw = tvec[2][0]  # Distance Forward to wall
    x_raw = tvec[0][0]  # Distance Lateral (Left/Right)

    gx, gy = None, None

    # --- WALL SPECIFIC LOGIC ---
    if wall == 'NORTH':
        # Camera facing North (Tag 0)
        # Global Y = Wall Y (2.0) - Distance Forward
        # Global X = Wall X (1.0) - Lateral Offset
        gy = ty - z_raw
        gx = tx - x_raw 

    elif wall == 'WEST':
        # Camera facing West (Tag 1)
        # Global X = Wall X (0.0) + Distance Forward
        # Global Y = Wall Y (1.0) - Lateral Offset (Check sign!)
        gx = tx + z_raw
        gy = ty - x_raw

    elif wall == 'EAST':
        # Camera facing East (Tag 2)
        # Global X = Wall X (2.0) - Distance Forward
        # Global Y = Wall Y (1.0) + Lateral Offset (Sign flips vs West)
        gx = tx - z_raw
        gy = ty + x_raw

    return gx, gy

def draw_minimap(frame, grid_coords, active_tag_id):
    """Draws the 2x2 grid, the car, and the tags."""
    h, w, _ = frame.shape
    map_size = 200
    margin = 20
    
    # Create Canvas
    map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255
    cell_px = map_size / GRID_COLS

    # 1. Draw Grid Lines
    for i in range(GRID_COLS + 1):
        x = int(i * cell_px)
        cv2.line(map_img, (x, 0), (x, map_size), (220, 220, 220), 1)
    for i in range(GRID_ROWS + 1):
        y = int(i * cell_px)
        cv2.line(map_img, (0, y), (map_size, y), (220, 220, 220), 1)

    # 2. Draw All Tags (as Black Rectangles)
    for tid, config in TAG_LIBRARY.items():
        # Scale Global Meters to Map Pixels
        # Map X = (Global X / 2.0) * map_size
        # Map Y = ((2.0 - Global Y) / 2.0) * map_size (Inverted Y)
        
        tx_px = int((config['x'] / GRID_WIDTH) * map_size)
        ty_px = int(((GRID_HEIGHT - config['y']) / GRID_HEIGHT) * map_size)
        
        color = (0, 0, 0) # Default Black
        if tid == active_tag_id:
            color = (0, 255, 0) # Highlight Active Tag Green
            
        cv2.rectangle(map_img, (tx_px-5, ty_px-5), (tx_px+5, ty_px+5), color, -1)
        # Draw ID
        cv2.putText(map_img, str(tid), (tx_px-15, ty_px-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    # 3. Draw Car (Red Dot)
    if grid_coords:
        r, c = grid_coords
        
        # Invert Row for Drawing (Row 0 is bottom)
        draw_row = (GRID_ROWS - 1) - r
        
        cx = int((c + 0.5) * cell_px)
        cy = int((draw_row + 0.5) * cell_px)
        
        cv2.circle(map_img, (cx, cy), 6, (0, 0, 255), -1)
        cv2.circle(map_img, (cx, cy), 8, (0, 0, 255), 1)

    # Overlay on Frame
    y_off = h - map_size - margin
    x_off = w - map_size - margin
    frame[y_off:y_off+map_size, x_off:x_off+map_size] = map_img
    cv2.rectangle(frame, (x_off, y_off), (x_off+map_size, y_off+map_size), (0,0,0), 2)
    cv2.putText(frame, "LIVE TRACKER", (x_off, y_off - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

def run_multi_tag_simulation():
    cap = cv2.VideoCapture(0)
    
    # Calibration
    ret, frame = cap.read()
    if not ret: return
    h, w = frame.shape[:2]
    focal_length = w * 0.8  
    cam_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1))
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    
    obj_points = np.array([[-TAG_SIZE/2, TAG_SIZE/2, 0], [TAG_SIZE/2, TAG_SIZE/2, 0], 
                           [TAG_SIZE/2, -TAG_SIZE/2, 0], [-TAG_SIZE/2, -TAG_SIZE/2, 0]], dtype=np.float32)

    print("Multi-Tag Grid System Active.")
    print("Tags: 0=North, 1=West, 2=East")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        grid_pos = None
        active_id = -1
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Find closest tag (smallest Z distance)
            best_dist = 999
            best_idx = -1
            
            # 1. Solve for all tags to find the closest one
            for i in range(len(ids)):
                tid = ids[i][0]
                if tid in TAG_LIBRARY:
                    success, _, tvec_check = cv2.solvePnP(obj_points, corners[i][0], cam_matrix, dist_coeffs)
                    if success:
                        dist = tvec_check[2][0]
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = i
            
            # 2. Use the closest tag for localization
            if best_idx != -1:
                active_id = ids[best_idx][0]
                success, rvec, tvec = cv2.solvePnP(obj_points, corners[best_idx][0], cam_matrix, dist_coeffs)
                
                if success:
                    gx, gy = calculate_global_pose(active_id, tvec)
                    
                    if gx is not None:
                        grid_pos = get_grid_coords(gx, gy)
                        
                        # --- DISPLAY ---
                        lines = [
                            f"LOCKED ON: Tag {active_id} ({TAG_LIBRARY[active_id]['wall']})",
                            f"Pos: ({gx:.2f}, {gy:.2f})",
                            f"Grid Cell: {grid_pos}",
                            f"Dist: {tvec[2][0]:.2f}m"
                        ]
                        
                        color = (0, 255, 0)
                        if gy > 2.0 or gx > 2.0 or gx < 0 or gy < 0:
                            color = (0, 0, 255) # Red if OOB
                            
                        for j, line in enumerate(lines):
                            cv2.putText(frame, line, (10, 40 + j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
                        cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, 0.1)

        draw_minimap(frame, grid_pos, active_id)

        cv2.imshow("Multi-Tag Localization", frame)
        if cv2.waitKey(1) == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_multi_tag_simulation()
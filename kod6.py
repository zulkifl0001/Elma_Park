import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import math
import heapq
import numpy as np
import random
import tkinter as tk
from tkinter import simpledialog

# --- 1. FİZİKSEL AYARLAR (CM) ---
ALAN_BOYUTU_CM = 200       
PARK_DERINLIK_CM = 30.0
PARK_GENISLIK_CM = 20.0
ARAC_UZUNLUK_CM = 26.0     
ARAC_GENISLIK_CM = 14.0    
AKS_MESAFESI_CM = 22.0     
DONUS_YARICAPI_CM = 40.0
ENGEL_MESAFESI_CM = 45.0   

# Güvenlik
arac_yaricapi = math.sqrt((ARAC_UZUNLUK_CM/2)**2 + (ARAC_GENISLIK_CM/2)**2) 
guvenlik_payi = 5.0 

# --- 2. PARK YERLERİ ---
PARK_YERLERI = {}
for i, y_cm in enumerate([50, 100, 150], 1):
    PARK_YERLERI[i] = {"center": (15.0, y_cm), "angle": 0, "size": (PARK_DERINLIK_CM, PARK_GENISLIK_CM), "wall": "left"}
for i, x_cm in enumerate([50, 100, 150], 4):
    PARK_YERLERI[i] = {"center": (x_cm, 185.0), "angle": -math.pi/2, "size": (PARK_GENISLIK_CM, PARK_DERINLIK_CM), "wall": "top"}
for i, y_cm in enumerate([150, 100, 50], 7):
    PARK_YERLERI[i] = {"center": (185.0, y_cm), "angle": math.pi, "size": (PARK_DERINLIK_CM, PARK_GENISLIK_CM), "wall": "right"}

# --- YARDIMCI FONKSİYONLAR ---
def get_switch_point(pid):
    spot = PARK_YERLERI[pid]
    cx, cy = spot["center"]
    r = DONUS_YARICAPI_CM * 1.3 
    offset = 15.0 
    if spot["wall"] == "left":   return (cx + r + offset, cy + r/2, math.pi/2) 
    elif spot["wall"] == "top":  return (cx + r/2, cy - r - offset, 0)
    elif spot["wall"] == "right": return (cx - r - offset, cy - r/2, -math.pi/2)
    return (100, 100, 0)

# --- MANUEL SENARYO OLUŞTURUCU (YENİ) ---
def setup_manual_scenario(pid):
    """
    Kullanıcının önce engelleri, sonra arabanın konumunu ve yönünü seçmesini sağlar.
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(0, ALAN_BOYUTU_CM)
    ax.set_ylim(0, ALAN_BOYUTU_CM)
    ax.grid(True, linestyle=':', alpha=0.4)
    
    # Park yerlerini çiz
    ax.add_patch(patches.Rectangle((-10,0), 10, 200, color='gray'))
    ax.add_patch(patches.Rectangle((200,0), 10, 200, color='gray'))
    ax.add_patch(patches.Rectangle((0,200), 200, 10, color='gray'))

    for i, data in PARK_YERLERI.items():
        cx, cy = data["center"]
        w, h = data["size"] 
        color = 'lightgreen' if i==pid else '#e0e0e0' 
        rect = patches.Rectangle((cx-w/2, cy-h/2), w, h, linewidth=2, edgecolor='gray', facecolor=color, alpha=0.6)
        ax.add_patch(rect)
        ax.text(cx, cy, f"P{i}", ha='center', va='center')
    
    # --- ADIM 1: ENGELLER ---
    obstacles = []
    ax.set_title("ADIM 1: 3 Adet Engel Yerleştirin (Tıklayın)", color='red', weight='bold')
    print("Haritaya tıklayarak 3 engel ekleyin...")
    
    for i in range(3):
        while True:
            pts = plt.ginput(1, timeout=-1, show_clicks=True)
            if not pts: return [], None # Çıkış
            cx, cy = pts[0]
            
            # Park yerlerinin üzerine engel koymayı önle
            collision = False
            for pid_check, spot in PARK_YERLERI.items():
                if math.sqrt((cx - spot["center"][0])**2 + (cy - spot["center"][1])**2) < 30.0:
                    collision = True
            
            if not collision:
                r_size = random.uniform(12.0, 18.0)
                obstacles.append((cx, cy, r_size))
                ax.add_patch(patches.Rectangle((cx - r_size/2, cy - r_size/2), r_size, r_size, color='black'))
                ax.text(cx, cy + 10, f"E{i+1}", color='red', ha='center')
                plt.draw()
                print(f"Engel {i+1} eklendi.")
                break
            else:
                print("HATA: Engel park yerine çok yakın!")

    # --- ADIM 2: ARABA KONUMU ---
    ax.set_title("ADIM 2: Arabanın MERKEZİNE Tıklayın", color='blue', weight='bold')
    print("Arabanın konumunu seçin...")
    
    car_start_pos = None
    
    while True:
        pts = plt.ginput(1, timeout=-1)
        if not pts: return [], None
        sx, sy = pts[0]
        
        # Çarpışma kontrolü (Engellere veya Duvarlara)
        if not check_collision(sx, sy, obstacles):
            # Geçici olarak noktayı göster
            ax.plot(sx, sy, 'bo')
            plt.draw()
            break
        else:
            print("HATA: Araba engele veya duvara çarpıyor! Tekrar seçin.")
            ax.set_title("ÇARPIŞMA! Başka yere tıklayın.", color='red', weight='bold')
            plt.pause(1.0)
            ax.set_title("ADIM 2: Arabanın MERKEZİNE Tıklayın", color='blue', weight='bold')

    # --- ADIM 3: ARABA YÖNÜ ---
    ax.set_title("ADIM 3: Arabanın YÖNÜNÜ belirlemek için uzağa tıklayın", color='purple', weight='bold')
    print("Arabanın yönünü seçin...")
    
    pts2 = plt.ginput(1, timeout=-1)
    if not pts2: return [], None
    dx, dy = pts2[0]
    
    # Açıyı hesapla (atan2)
    syaw = math.atan2(dy - sy, dx - sx)
    car_start_pos = (sx, sy, syaw)
    
    # Görselleştirme (Ok çiz)
    ax.arrow(sx, sy, 20*math.cos(syaw), 20*math.sin(syaw), head_width=5, color='blue')
    plt.draw()
    plt.pause(1.0)
    plt.close(fig) # Pencereyi kapat, simülasyona geç
    
    return obstacles, car_start_pos


# --- CUBIC BEZIER ---
def generate_cubic_bezier_path(start_conf, end_conf, pid):
    sp_x, sp_y, sp_yaw = start_conf
    target_x, target_y = end_conf[0], end_conf[1]
    
    dist_start = 60.0 
    cp1_x = sp_x - dist_start * math.cos(sp_yaw)
    cp1_y = sp_y - dist_start * math.sin(sp_yaw)
    
    dist_end = 60.0 
    wall = PARK_YERLERI[pid]["wall"]
    
    if wall == "top":   
        cp2_x = target_x 
        cp2_y = target_y - dist_end 
    elif wall == "left": 
        cp2_x = target_x + dist_end 
        cp2_y = target_y
    elif wall == "right": 
        cp2_x = target_x - dist_end 
        cp2_y = target_y
    else:
        cp2_x, cp2_y = target_x, target_y

    path_elements = []
    steps = 60 
    
    for i in range(steps + 1):
        t = i / steps
        bx = (1-t)**3 * sp_x + 3*(1-t)**2 * t * cp1_x + 3*(1-t)*t**2 * cp2_x + t**3 * target_x
        by = (1-t)**3 * sp_y + 3*(1-t)**2 * t * cp1_y + 3*(1-t)*t**2 * cp2_y + t**3 * target_y
        
        if i < steps:
            t_next = (i + 0.05) / steps
            bx_next = (1-t_next)**3 * sp_x + 3*(1-t_next)**2 * t_next * cp1_x + 3*(1-t_next)*t_next**2 * cp2_x + t_next**3 * target_x
            by_next = (1-t_next)**3 * sp_y + 3*(1-t_next)**2 * t_next * cp1_y + 3*(1-t_next)*t_next**2 * cp2_y + t_next**3 * target_y
            
            dx = bx_next - bx
            dy = by_next - by
            move_angle = math.atan2(dy, dx)
            byaw = move_angle + math.pi 
        else:
            byaw = PARK_YERLERI[pid]["angle"]
            if PARK_YERLERI[pid]["wall"] == "right": byaw = -math.pi

        byaw = (byaw + math.pi) % (2 * math.pi) - math.pi
        path_elements.append((bx, by, byaw, -1, 0))
        
    return path_elements

# --- PLANLAMA ---
def move_vehicle(x, y, yaw, distance, steering):
    theta = yaw + (distance / AKS_MESAFESI_CM) * math.tan(steering)
    theta = (theta + math.pi) % (2 * math.pi) - math.pi 
    nx = x + distance * math.cos(theta)
    ny = y + distance * math.sin(theta)
    return nx, ny, theta

def check_collision(x, y, obstacles):
    if x < 15 or x > 185 or y < 15 or y > 185: return True
    for obs in obstacles:
        ox, oy, osize = obs
        engel_yaricapi = osize / 2 * math.sqrt(2)
        limit_mesafe = engel_yaricapi + arac_yaricapi + guvenlik_payi
        if math.sqrt((x-ox)**2 + (y-oy)**2) < limit_mesafe: return True
    for pid, spot in PARK_YERLERI.items():
        if math.sqrt((x - spot["center"][0])**2 + (y - spot["center"][1])**2) < 25.0: return True 
    return False

class Node:
    def __init__(self, x, y, yaw, g, h, parent=None, steering=0, direction=1):
        self.x, self.y, self.yaw = x, y, yaw
        self.g, self.h, self.f = g, h, g+h
        self.parent = parent
        self.steering = steering 
        self.direction = direction
    def __lt__(self, other): return self.f < other.f

def hybrid_a_star_kinematic(start, end, obstacles):
    start_node = Node(start[0], start[1], start[2], 0, 0, direction=1)
    open_list = []; heapq.heappush(open_list, start_node)
    closed = set()
    
    fixed_steer = math.atan(AKS_MESAFESI_CM / DONUS_YARICAPI_CM) 
    steer_actions = [-fixed_steer, 0, fixed_steer] 
    directions = [1, -1] 
    step_size = 10.0 
    max_iter = 50000 
    count = 0

    while open_list:
        count += 1
        if count > max_iter: return None
        curr = heapq.heappop(open_list)
        
        dist_to_target = math.sqrt((curr.x-end[0])**2 + (curr.y-end[1])**2)
        if dist_to_target < 15.0: 
            path = []
            while curr:
                path.append((curr.x, curr.y, curr.yaw, curr.direction, curr.steering))
                curr = curr.parent
            return path[::-1]
            
        grid_idx = (int(curr.x // 10), int(curr.y // 10), int(math.degrees(curr.yaw) // 15))
        if grid_idx in closed: continue
        closed.add(grid_idx)
        
        for direction in directions:
            for steer in steer_actions:
                move_dist = step_size * direction
                nx, ny, nyaw = move_vehicle(curr.x, curr.y, curr.yaw, move_dist, steer)
                if check_collision(nx, ny, obstacles): continue
                
                cost = step_size
                if direction == -1: cost += 10.0 
                if direction != curr.direction: cost += 20.0 
                steer_penalty = abs(steer - curr.steering) * 5.0
                
                new_g = curr.g + cost + steer_penalty
                new_h = math.sqrt((nx-end[0])**2 + (ny-end[1])**2)
                heapq.heappush(open_list, Node(nx, ny, nyaw, new_g, new_h, 
                                               parent=curr, steering=steer, direction=direction))
    return None

# --- GÖRSELLEŞTİRME VE ANİMASYON ---
def init_vehicle_drawing(ax):
    body = patches.Rectangle((0, 0), ARAC_UZUNLUK_CM, ARAC_GENISLIK_CM, 
                             facecolor='blue', edgecolor='black', alpha=0.8, zorder=10)
    ax.add_patch(body)
    w_len, w_wid = 6, 3
    fl_wheel = patches.Rectangle((0, 0), w_len, w_wid, color='black', zorder=11)
    fr_wheel = patches.Rectangle((0, 0), w_len, w_wid, color='black', zorder=11)
    rl_wheel = patches.Rectangle((0, 0), w_len, w_wid, color='black', zorder=11)
    rr_wheel = patches.Rectangle((0, 0), w_len, w_wid, color='black', zorder=11)
    ax.add_patch(fl_wheel); ax.add_patch(fr_wheel)
    ax.add_patch(rl_wheel); ax.add_patch(rr_wheel)
    return {'body': body, 'fl': fl_wheel, 'fr': fr_wheel, 'rl': rl_wheel, 'rr': rr_wheel}

def update_vehicle_drawing(car_graphics, x, y, yaw, steering, direction):
    color = 'blue' if direction == 1 else 'orange'
    if direction == -1 and abs(steering) < 0.1: color = '#9932CC' # Mor
    car_graphics['body'].set_facecolor(color)

    car_graphics['body'].set_angle(math.degrees(yaw))
    cx, cy = x, y
    l, w = ARAC_UZUNLUK_CM, ARAC_GENISLIK_CM
    corner_x = cx - (l/2)*math.cos(yaw) + (w/2)*math.sin(yaw)
    corner_y = cy - (l/2)*math.sin(yaw) - (w/2)*math.cos(yaw)
    car_graphics['body'].set_xy((corner_x, corner_y))
    
    def get_world_pos(lx, ly, car_x, car_y, car_yaw):
        wx = car_x + lx * math.cos(car_yaw) - ly * math.sin(car_yaw)
        wy = car_y + lx * math.sin(car_yaw) + ly * math.cos(car_yaw)
        return wx, wy

    w_len, w_wid = 6, 3
    wheel_angle = math.degrees(yaw + steering)
    
    fl_cx, fl_cy = get_world_pos(AKS_MESAFESI_CM/2, ARAC_GENISLIK_CM/2 - 1, x, y, yaw)
    car_graphics['fl'].set_angle(wheel_angle)
    car_graphics['fl'].set_xy((fl_cx - (w_len/2)*math.cos(yaw+steering) + (w_wid/2)*math.sin(yaw+steering),
                               fl_cy - (w_len/2)*math.sin(yaw+steering) - (w_wid/2)*math.cos(yaw+steering)))

    fr_cx, fr_cy = get_world_pos(AKS_MESAFESI_CM/2, -ARAC_GENISLIK_CM/2 + 1, x, y, yaw)
    car_graphics['fr'].set_angle(wheel_angle)
    car_graphics['fr'].set_xy((fr_cx - (w_len/2)*math.cos(yaw+steering) + (w_wid/2)*math.sin(yaw+steering),
                               fr_cy - (w_len/2)*math.sin(yaw+steering) - (w_wid/2)*math.cos(yaw+steering)))
    
    r_angle = math.degrees(yaw)
    rl_cx, rl_cy = get_world_pos(-AKS_MESAFESI_CM/2, ARAC_GENISLIK_CM/2 - 1, x, y, yaw)
    car_graphics['rl'].set_angle(r_angle)
    car_graphics['rl'].set_xy((rl_cx - (w_len/2)*math.cos(yaw) + (w_wid/2)*math.sin(yaw),
                               rl_cy - (w_len/2)*math.sin(yaw) - (w_wid/2)*math.cos(yaw)))

    rr_cx, rr_cy = get_world_pos(-AKS_MESAFESI_CM/2, -ARAC_GENISLIK_CM/2 + 1, x, y, yaw)
    car_graphics['rr'].set_angle(r_angle)
    car_graphics['rr'].set_xy((rr_cx - (w_len/2)*math.cos(yaw) + (w_wid/2)*math.sin(yaw),
                               rr_cy - (w_len/2)*math.sin(yaw) - (w_wid/2)*math.cos(yaw)))

def run_simulation():
    # 1. Park Yeri Seçimi
    root = tk.Tk(); root.withdraw()
    pid = simpledialog.askinteger("Otonom Park", "Hedef Park (1-9):", minvalue=1, maxvalue=9)
    if not pid: return

    spot = PARK_YERLERI[pid]
    target_pos = spot["center"]
    ideal_switch_point = get_switch_point(pid)
    
    # 2. TAM MANUEL SENARYO OLUŞTURMA (Engeller + Araba)
    obstacles, start_pos = setup_manual_scenario(pid)
    
    if not obstacles or not start_pos:
        print("Senaryo oluşturulmadı veya iptal edildi.")
        return

    print(f"Başlangıç: {int(start_pos[0])}, {int(start_pos[1])} | Açı: {math.degrees(start_pos[2]):.1f}")
    print("Cubic Bezier ile hesaplanıyor...")
    
    path_to_switch = hybrid_a_star_kinematic(start_pos, ideal_switch_point, obstacles)
    
    full_path = []
    
    if path_to_switch:
        full_path.extend(path_to_switch)
        actual_switch_state = path_to_switch[-1] 
        actual_switch_conf = (actual_switch_state[0], actual_switch_state[1], actual_switch_state[2])
        bezier_path = generate_cubic_bezier_path(actual_switch_conf, target_pos, pid)
        full_path.extend(bezier_path)
    else:
        print("HATA: Seçtiğiniz konumdan bir rota bulunamadı!")
        # Rota bulunamasa bile durumu göstermek için boş haritayı açabiliriz
    
    # --- GRAFİK ---
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(0, ALAN_BOYUTU_CM) 
    ax.set_ylim(0, ALAN_BOYUTU_CM)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_title(f"Park {pid}: Tam Manuel Senaryo", fontsize=12, weight='bold')

    ax.add_patch(patches.Rectangle((-10,0), 10, 200, color='gray'))
    ax.add_patch(patches.Rectangle((200,0), 10, 200, color='gray'))
    ax.add_patch(patches.Rectangle((0,200), 200, 10, color='gray'))

    for i, data in PARK_YERLERI.items():
        cx, cy = data["center"]
        w, h = data["size"] 
        color = 'lightgreen' if i==pid else '#e0e0e0' 
        rect = patches.Rectangle((cx-w/2, cy-h/2), w, h, linewidth=2, edgecolor='gray', facecolor=color, alpha=0.6)
        ax.add_patch(rect)
        ax.text(cx, cy, f"P{i}", ha='center', va='center')

    # Engelleri Çiz
    for obs in obstacles:
        ox, oy, osize = obs
        ax.add_patch(patches.Rectangle((ox - osize/2, oy - osize/2), osize, osize, color='#333'))
        engel_yaricapi = osize / 2 * math.sqrt(2)
        ax.add_patch(plt.Circle((ox, oy), engel_yaricapi + arac_yaricapi + guvenlik_payi, color='red', alpha=0.1, linestyle='--'))

    if full_path:
        px = [p[0] for p in full_path]
        py = [p[1] for p in full_path]
        ax.plot(px, py, 'k--', alpha=0.3, linewidth=1)
        
        car_graphics = init_vehicle_drawing(ax)
        info_text = ax.text(5, 5, "", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        def animate(frame_idx):
            if frame_idx >= len(full_path):
                frame_idx = len(full_path) - 1
            x, y, yaw, direction, steering = full_path[frame_idx]
            update_vehicle_drawing(car_graphics, x, y, yaw, steering, direction)
            mod = "İLERİ" if direction == 1 else "GERİ"
            if direction == -1 and abs(steering) < 0.1: mod = "PARK"
            info_text.set_text(f"Durum: {mod}")
            return car_graphics.values()

        ani = animation.FuncAnimation(fig, animate, frames=len(full_path)+20, interval=40, blit=False, repeat=True)
        plt.show()
    else:
        # Rota bulunamadıysa sadece başlangıç noktasını çizip göster
        car_graphics = init_vehicle_drawing(ax)
        update_vehicle_drawing(car_graphics, start_pos[0], start_pos[1], start_pos[2], 0, 1)
        ax.text(100, 100, "ROTA BULUNAMADI", color='red', fontsize=16, ha='center', weight='bold')
        plt.show()

if __name__ == "__main__":
    run_simulation()
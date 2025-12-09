# --- Interactive Calibration Tool: Align Trajectory to Online Map ---
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path
from rosbags.highlevel import AnyReader
from pyproj import Transformer
import contextily as cx

# =========================================================
# ⚙️ USER SETTINGS
# =========================================================
bag_path      = Path('/media/davide/New Volume/07-11-2025/2025-11-07-11-34-46_fixed.bag')
centroid_file = "/home/davide/catkin_ws/src/lidar_test/scripts/yolo_centroids_clusters_orient_cls.txt"
odom_topic    = "/odom"
buffer_m      = 150  # Map zoom buffer (meters)

# --- SLIDER RANGES ---
SHIFT_RANGE   = 50.0 
YAW_CENTER    = -0.04000000
YAW_RANGE     = 0.1  # Allows tweaking +/- 0.1 radians

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
LAT0 = 48.17552100
LON0 = 11.59523900
ODOM0_X = 692925.990
ODOM0_Y = 5339070.997

# =========================================================
# 1. LOAD DATA (Odom + YOLO)
# =========================================================
print("⏳ Loading Rosbag...")
with AnyReader([bag_path]) as reader:
    conns = [x for x in reader.connections if x.topic == odom_topic]
    if not conns: raise ValueError("Odom topic not found!")
    
    odoms = []
    for c in conns:
        for connection, ts, raw in reader.messages(connections=[c]):
            m = reader.deserialize(raw, connection.msgtype)
            p = m.pose.pose.position
            odoms.append((p.x, p.y))

odom_data = np.array(odoms)
odom_x_raw = odom_data[:, 0]
odom_y_raw = odom_data[:, 1]
print(f"✅ Loaded {len(odom_x_raw)} Odom points.")

# Load Centroids
print("⏳ Loading Centroids...")
cent_x_raw, cent_y_raw = np.array([]), np.array([])
if os.path.exists(centroid_file):
    try:
        raw = np.genfromtxt(centroid_file, delimiter=",", names=True, dtype=None, encoding=None, invalid_raise=False)
        if raw.dtype.names and "x_world" in raw.dtype.names:
            cent_x_raw = np.asarray(raw["x_world"], dtype=float)
            cent_y_raw = np.asarray(raw["y_world"], dtype=float)
        else:
            raw_np = np.genfromtxt(centroid_file, delimiter=",", invalid_raise=False)
            if raw_np.ndim == 2:
                if np.all(np.mod(raw_np[:,0], 1) == 0): 
                    cent_x_raw, cent_y_raw = raw_np[:,1], raw_np[:,2]
                else:
                    cent_x_raw, cent_y_raw = raw_np[:,0], raw_np[:,1]
    except Exception as e:
        print(f"⚠️ Could not load centroids: {e}")
print(f"✅ Loaded {len(cent_x_raw)} Centroids.")

# =========================================================
# 2. PROJECTION LOGIC (Real-time)
# =========================================================
transformer_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def enu_to_latlon_fast(dx, dy):
    lat_ref_rad = math.radians(LAT0)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2*lat_ref_rad)
    m_per_deg_lon = 111412.84 * math.cos(lat_ref_rad)
    return LAT0 + (dy / m_per_deg_lat), LON0 + (dx / m_per_deg_lon)

def get_projected_coords(x_arr, y_arr, shift_x, shift_y, yaw_offset):
    if len(x_arr) == 0: return np.array([]), np.array([])
    
    # 1. Center & Apply Shift (Translation)
    dx = (x_arr - ODOM0_X) + shift_x
    dy = (y_arr - ODOM0_Y) + shift_y
    
    # 2. Apply Rotation (Interactive Yaw)
    c, s = math.cos(yaw_offset), math.sin(yaw_offset)
    dx_e = c*dx - s*dy
    dy_n = s*dx + c*dy
    
    # 3. Convert to LatLon
    lat, lon = enu_to_latlon_fast(dx_e, dy_n)
    
    # 4. Project to WebMercator
    xm, ym = transformer_to_3857.transform(lon, lat)
    return xm, ym

# =========================================================
# 3. INTERACTIVE PLOT SETUP
# =========================================================
print("🎨 Preparing Plot...")
init_xm, init_ym = get_projected_coords(odom_x_raw, odom_y_raw, 0, 0, YAW_CENTER)
init_cx, init_cy = get_projected_coords(cent_x_raw, cent_y_raw, 0, 0, YAW_CENTER)

xmin, xmax = np.min(init_xm) - buffer_m, np.max(init_xm) + buffer_m
ymin, ymax = np.min(init_ym) - buffer_m, np.max(init_ym) + buffer_m

fig, ax = plt.subplots(figsize=(12, 10))
plt.subplots_adjust(bottom=0.30) # More space for sliders

# 1. Add Background Map
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
try:
    provider = cx.providers.CartoDB.Positron
    cx.add_basemap(ax, crs="EPSG:3857", source=provider)
    print("✅ Online Map Background Loaded.")
except Exception as e:
    print(f"⚠️ Failed to load map tiles: {e}")

# 2. Plot Data
traj_line, = ax.plot(init_xm, init_ym, color='red', linewidth=1.5, alpha=0.8, label='Trajectory', zorder=10)
cent_scat = ax.scatter(init_cx, init_cy, s=10, color='limegreen', edgecolor='k', linewidth=0.3, label='Centroids', zorder=11)

# --- FIXED LINE BELOW ---
ax.set_title("Calibration Tool: Adjust X, Y, and Rotation")
# ------------------------

ax.grid(False) 
ax.legend(loc='upper right')

# =========================================================
# 4. SLIDERS & LOGIC
# =========================================================
ax_sx = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_sy = plt.axes([0.2, 0.10, 0.65, 0.03])
ax_yaw = plt.axes([0.2, 0.05, 0.65, 0.03]) 

s_x = Slider(ax_sx, 'Shift X (East)', -SHIFT_RANGE, SHIFT_RANGE, valinit=0.0)
s_y = Slider(ax_sy, 'Shift Y (North)', -SHIFT_RANGE, SHIFT_RANGE, valinit=0.0)
s_yaw = Slider(ax_yaw, 'Yaw Offset', YAW_CENTER - YAW_RANGE, YAW_CENTER + YAW_RANGE, valinit=YAW_CENTER)

def update(val):
    sx = s_x.val
    sy = s_y.val
    yaw = s_yaw.val
    
    nx, ny = get_projected_coords(odom_x_raw, odom_y_raw, sx, sy, yaw)
    traj_line.set_data(nx, ny)
    
    if len(cent_x_raw) > 0:
        ncx, ncy = get_projected_coords(cent_x_raw, cent_y_raw, sx, sy, yaw)
        cent_scat.set_offsets(np.c_[ncx, ncy])
        
    fig.canvas.draw_idle()

s_x.on_changed(update)
s_y.on_changed(update)
s_yaw.on_changed(update)

reset_ax = plt.axes([0.8, 0.01, 0.1, 0.04])
button = Button(reset_ax, 'Print Values', hovercolor='0.975')

def print_values(event):
    print("\n" + "="*40)
    print(f"🎯 YOUR CALIBRATION VALUES:")
    print(f"SHIFT_X    = {s_x.val:.3f}")
    print(f"SHIFT_Y    = {s_y.val:.3f}")
    print(f"YAW_OFFSET = {s_yaw.val:.8f}")
    print("="*40 + "\n")

button.on_clicked(print_values)

print("🚀 Launching Interface...")
plt.show()
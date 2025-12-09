# --- Overlay Apollo odom trajectory + YOLO centroids on OSM + steering plots ---
# Requires:
#   Common: pip install numpy matplotlib rosbags geographiclib
#   For .pbf maps: pip install pyrosm geopandas shapely pyproj
#   For .osm maps: pip install osmnx geopandas shapely pyproj
#   Online tiles: pip install contextily xyzservices
# Note: GeoPandas/Shapely require system deps on some platforms.

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader

import geopandas as gpd
from shapely.geometry import LineString, box
from pyproj import Transformer
import contextily as cx
from matplotlib.lines import Line2D

# --------------------- USER SETTINGS ---------------------
bag_path   = Path('/media/davide/New Volume/07-11-2025/2025-11-07-11-34-46_fixed.bag')
odom_topic = "/odom"

# Float topics (e.g., commands/feedback)
cmd_topic      = "/cmd/steering_target"
feedback_topic = "/vehicle/steering_pct"
float_topics   = [cmd_topic, feedback_topic]

# Offline OSM file (.pbf or .osm)
osm_path = "/home/davide/catkin_ws/src/lidar_test/scripts/map.osm"  # or .pbf

# File centroidi YOLO (coordinate nel frame odom/world: cluster_id, x_world, y_world, ...)
centroid_file = "yolo_centroids_clusters_orient_cls.txt"

# Margine mappa attorno al percorso (metri)
buffer_m = 200
# --------------------------------------------------------

# --- Hardcoded Apollo odom XY → WGS84 converter (use your measured constants) ---
LAT0 = 48.17552100
LON0 = 11.59523900
ALT0 = 0.000
ODOM0_X = 692925.990
ODOM0_Y = 5339070.997
YAW_OFFSET = -0.03000000  # radians (CCW positive)

# Optional: precise geodesic if available, else fallback
try:
    from geographiclib.geodesic import Geodesic
    _USE_GEODESIC = True
except Exception:
    Geodesic = None
    _USE_GEODESIC = False


def enu_to_latlon(dx_m, dy_m, lat_ref, lon_ref):
    """Convert local ENU offsets (east, north) to WGS84 lat/lon."""
    if _USE_GEODESIC:
        dist = float(math.hypot(dx_m, dy_m))
        azi_deg = math.degrees(math.atan2(dx_m, dy_m))  # east-of-north
        g = Geodesic.WGS84.Direct(lat_ref, lon_ref, azi_deg, dist)
        return g['lat2'], g['lon2']
    # Small-area approximation
    lat_ref_rad = math.radians(lat_ref)
    m_per_deg_lat = (
        111132.92
        - 559.82 * math.cos(2 * lat_ref_rad)
        + 1.175 * math.cos(4 * lat_ref_rad)
    )
    m_per_deg_lon = (
        111412.84 * math.cos(lat_ref_rad)
        - 93.5 * math.cos(3 * lat_ref_rad)
    )
    dlat_deg = dy_m / m_per_deg_lat
    dlon_deg = dx_m / m_per_deg_lon
    return lat_ref + dlat_deg, lon_ref + dlon_deg


def odom_xy_to_wgs84_vec(x_arr: np.ndarray, y_arr: np.ndarray):
    """Vectorized conversion from Apollo odom XY to (lat, lon) using hardcoded reference."""
    dx = x_arr.astype(float) - ODOM0_X
    dy = y_arr.astype(float) - ODOM0_Y
    c, s = math.cos(YAW_OFFSET), math.sin(YAW_OFFSET)
    dx_e = c * dx - s * dy   # east
    dy_n = s * dx + c * dy   # north
    lat = np.empty_like(dx_e, dtype=float)
    lon = np.empty_like(dy_n, dtype=float)
    for i in range(dx_e.size):
        lat[i], lon[i] = enu_to_latlon(dx_e[i], dy_n[i], LAT0, LON0)
    return lat, lon


# -------------- Read bag: /odom + float topics --------------
def quat_to_yaw(x, y, z, w):
    s = 2.0 * (w*z + x*y)
    c = 1.0 - 2.0 * (y*y + z*z)
    return math.atan2(s, c)


with AnyReader([bag_path]) as reader:
    conns = {}
    for c in reader.connections:
        conns.setdefault(c.topic, []).append(c)

    # Sanity checks
    for t in [odom_topic, *float_topics]:
        if t not in conns:
            raise ValueError(f"Topic not found in bag: {t}")

    # Gather odometry (t, x, y, yaw)
    odoms = []
    for c in conns[odom_topic]:
        for connection, ts, raw in reader.messages(connections=[c]):
            m = reader.deserialize(raw, connection.msgtype)
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
            odoms.append((ts * 1e-9, float(p.x), float(p.y), yaw))

    # Gather float topics
    floats = {t: [] for t in float_topics}
    for t in float_topics:
        for c in conns[t]:
            for connection, ts, raw in reader.messages(connections=[c]):
                msg = reader.deserialize(raw, connection.msgtype)
                val = getattr(msg, 'data', getattr(msg, 'value', None))
                if val is not None:
                    floats[t].append((ts * 1e-9, float(val)))

# Sort + build arrays with common t0
odoms.sort(key=lambda x: x[0])
for t in float_topics:
    floats[t].sort(key=lambda x: x[0])

if not odoms:
    raise RuntimeError("No odometry messages found to compute yaw/pose.")

t0 = min([odoms[0][0]] + [vals[0][0] for vals in floats.values() if vals]) if floats else odoms[0][0]

# -------------------- DEBUG: raw odometry before transformation --------------------
odom_x = np.array([x for (_, x, _, _) in odoms], dtype=float)
odom_y = np.array([y for (_, _, y, _) in odoms], dtype=float)

print("\n=== RAW ODOMETRY (before transform) ===")
print(f"Samples: {odom_x.size}")
print(f"X range: {np.min(odom_x):.3f}  →  {np.max(odom_x):.3f}")
print(f"Y range: {np.min(odom_y):.3f}  →  {np.max(odom_y):.3f}")
print(f"First 3 odom points:")
for i in range(min(3, odom_x.size)):
    print(f"  ({odom_x[i]:.3f}, {odom_y[i]:.3f})")
print("========================================\n")

odom_t   = np.array([t - t0 for (t, _, _, _) in odoms], dtype=float)
odom_x   = np.array([x for (_, x, _, _) in odoms], dtype=float)
odom_y   = np.array([y for (_, _, y, _) in odoms], dtype=float)
odom_yaw = np.array([yaw for (_, _, _, yaw) in odoms], dtype=float)
float_ts   = {k: np.array([t - t0 for t, _ in v], dtype=float) for k, v in floats.items()}
float_vals = {k: np.array([x for _, x in v], dtype=float) for k, v in floats.items()}

# -------------- Convert odom XY → lat/lon (trajectory) --------------
lat, lon = odom_xy_to_wgs84_vec(odom_x, odom_y)

# ================= LOAD + TRANSFORM YOLO CENTROIDS (with orientation) =================
def load_centroids_with_orientation(path):
    """
    Ritorna (cluster_id, x, y, orient) dal file di centroidi.

    Supporta:
    - .npy: nessuna orientazione -> "unknown"
    - .txt/.csv:
        1, 692934.742, 5339060.430, 549.472, 65, 0.901, parallel
        2, 692940.161, 5339057.322, 549.572, 138, 0.898, perpendicular
      oppure senza orientazione (ultima colonna numerica).
    """
    if not os.path.exists(path):
        print(f"[WARN] Centroid file not found: {path}")
        return np.array([]), np.array([]), np.array([]), [] # Aggiunto array vuoto per l'ID

    ext = os.path.splitext(path)[1].lower()


    # testo (.txt / .csv)
    ids, xs, ys, orients = [], [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue

            # preferisci formato [cluster_id, x, y, ...]
            id_val = x_val = y_val = None
            try:
                # Prova a leggere l'ID (colonna 0), X (colonna 1), Y (colonna 2)
                id_val = int(parts[0])
                x_val = float(parts[1])
                y_val = float(parts[2])
            except (IndexError, ValueError):
                # fallback: [x, y, ...]
                try:
                    # ID fittizio (sequenziale)
                    id_val = len(ids) + 1 
                    x_val = float(parts[0])
                    y_val = float(parts[1])
                except (IndexError, ValueError):
                    continue

            # Orientamento è l'ultimo token
            last_token = parts[6].lower()
            if last_token in ("parallel", "perpendicular", "unknown"):
                o_str = last_token
            else:
                o_str = "unknown"

            ids.append(id_val)
            xs.append(x_val)
            ys.append(y_val)
            orients.append(o_str)

    if not xs:
        print(f"[WARN] No valid centroid lines parsed from {path}")
        return np.array([]), np.array([]), np.array([]), []

    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)
    id_arr = np.array(ids, dtype=int)
    return id_arr, x_arr, y_arr, orients # Restituisce ID, x, y, orient


cent_id, cent_x, cent_y, cent_orient = load_centroids_with_orientation(centroid_file)
if cent_x.size:
    print(f"Loaded {cent_x.size} centroids from {centroid_file}")
    print("Centroids ODOM X range:", np.min(cent_x), np.max(cent_x))
    print("Centroids ODOM Y range:", np.min(cent_y), np.max(cent_y))
else:
    print("[INFO] No centroids loaded.")

# ------------ Costruisci segmenti orientati in ODOM (barrette) ------------
# cent_segments_odom è ora una lista di (x_start, y_start, x_end, y_end, label, cluster_id)
cent_segments_odom = [] 

if cent_x.size:
    dash_len = 4.0  # lunghezza barretta in metri (approx lunghezza auto)
    N_odom = odom_x.size

    for i in range(cent_x.size):
        cx = cent_x[i]
        cy = cent_y[i]
        c_id = cent_id[i] # Aggiunta l'acquisizione dell'ID
        label = cent_orient[i] if i < len(cent_orient) else "unknown"
        lab = str(label).lower()

        # trova il punto /odom più vicino al centroido
        d2 = (odom_x - cx) ** 2 + (odom_y - cy) ** 2
        idx = int(np.argmin(d2))

        if N_odom < 2:
            continue

        # heading della traiettoria vicino al centroido (in odom)
        if idx == 0:
            x0, y0 = odom_x[0], odom_y[0]
            x1, y1 = odom_x[1], odom_y[1]
        elif idx == N_odom - 1:
            x0, y0 = odom_x[-2], odom_y[-2]
            x1, y1 = odom_x[-1], odom_y[-1]
        else:
            x0, y0 = odom_x[idx - 1], odom_y[idx - 1]
            x1, y1 = odom_x[idx + 1], odom_y[idx + 1]

        heading = math.atan2(y1 - y0, x1 - x0)  # direzione strada ~ traiettoria ego

        # parallel → stessa direzione della strada
        # perpendicular → +90°
        if lab == "perpendicular":
            heading += math.pi / 2.0

        dx = 0.5 * dash_len * math.cos(heading)
        dy = 0.5 * dash_len * math.sin(heading)

        x_start = cx - dx
        y_start = cy - dy
        x_end   = cx + dx
        y_end   = cy + dy

        # Inserimento dell'ID nella lista dei segmenti
        cent_segments_odom.append((x_start, y_start, x_end, y_end, lab, c_id))

# -------------- Convert centroids to lat/lon (same transform) --------------
if cent_x.size:
    cent_lat, cent_lon = odom_xy_to_wgs84_vec(cent_x, cent_y)
else:
    cent_lat = cent_lon = np.array([])

# -------------- Prepare OSM overlay --------------
# 1) Project trajectory to Web Mercator (EPSG:3857)
transformer_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)  # lon, lat
x_merc, y_merc = transformer_to_3857.transform(lon.astype(float), lat.astype(float))

# 2) Compute plotting bounds in EPSG:3857 (meters)
xmin, xmax = float(np.min(x_merc)), float(np.max(x_merc))
ymin, ymax = float(np.min(y_merc)), float(np.max(y_merc))
xmin -= buffer_m; ymin -= buffer_m
xmax += buffer_m; ymax += buffer_m

# Helper: meters -> degrees for bbox in lat/lon
lon0_track, lat0_track = float(lon[0]), float(lat[0])
lat0_rad = math.radians(lat0_track)
m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat0_rad) + 1.175 * math.cos(4 * lat0_rad)
m_per_deg_lon = 111412.84 * math.cos(lat0_rad) - 93.5 * math.cos(3 * lat0_rad)
lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
lat_pad = buffer_m / m_per_deg_lat
lon_pad = buffer_m / m_per_deg_lon
bbox_latlon = (lat_min - lat_pad, lon_min - lon_pad, lat_max + lat_pad, lon_max + lon_pad)

# 3) Load ways/roads from the OSM file
ext = os.path.splitext(osm_path)[1].lower()

roads3857 = None
if ext == ".pbf":
    try:
        from pyrosm import OSM
    except Exception as e:
        raise RuntimeError("pyrosm not installed. `pip install pyrosm`") from e

    min_lat, min_lon, max_lat, max_lon = bbox_latlon
    osm = OSM(osm_path, bounding_box=(min_lon, min_lat, max_lon, max_lat))  # (lon, lat)
    roads = osm.get_network(network_type="driving")  # or "all"
    if roads is None or roads.empty:
        raise RuntimeError("No roads found in bbox from the PBF. Enlarge buffer_m or check file area.")
    roads3857 = roads.to_crs(3857)

elif ext == ".osm":
    try:
        import osmnx as ox
    except Exception as e:
        raise RuntimeError("osmnx not installed. `pip install osmnx`") from e

    G = ox.graph_from_xml(osm_path, simplify=True)
    gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
    if gdf.empty:
        raise RuntimeError("No edges found in the .osm XML.")
    roads3857 = gdf.to_crs(3857)
    clip_poly = box(xmin, ymin, xmax, ymax)
    roads3857 = gpd.clip(roads3857, clip_poly)
else:
    raise ValueError(f"Unsupported OSM file extension: {ext}. Use .pbf (preferred) or .osm")

# 4) Plot OFFLINE: roads + trajectory + oriented centroids
traj_line = gpd.GeoSeries([LineString(np.column_stack([x_merc, y_merc]))], crs="EPSG:3857")

fig, ax = plt.subplots(figsize=(10, 10)) # Dimensione aumentata
roads3857.plot(ax=ax, linewidth=0.8, color='gray', alpha=0.7)
traj_line.plot(ax=ax, linewidth=2.0, color="red", label="Trajectory")

# --- Modifica: Aggiungi ID al plot Offline OSM ---
if cent_segments_odom:
    for (x_s, y_s, x_e, y_e, lab, c_id) in cent_segments_odom:
        lat_seg, lon_seg = odom_xy_to_wgs84_vec(
            np.array([x_s, x_e]), np.array([y_s, y_e])
        )
        xs_merc, ys_merc = transformer_to_3857.transform(
            lon_seg.astype(float), lat_seg.astype(float)
        )

        if lab == "parallel":
            color = "limegreen"
        elif lab == "perpendicular":
            color = "magenta"
        else:
            color = "yellow"

        # Disegna il dash
        ax.plot(xs_merc, ys_merc, linewidth=1.5, color=color, alpha=0.9, zorder=5)
        
        # Aggiungi il testo ID (posizione: punto centrale del dash)
        cx_merc = (xs_merc[0] + xs_merc[1]) / 2
        cy_merc = (ys_merc[0] + ys_merc[1]) / 2
        
        ax.text(
            cx_merc, 
            cy_merc, 
            f"ID:{c_id}", 
            color='black', 
            fontsize=8, 
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5), 
            ha='center', 
            va='center',
            zorder=6
        )

# legenda custom per le barrette
legend_elements = [
    Line2D([0], [0], color='limegreen', lw=1.5, label='YOLO parallel'),
    Line2D([0], [0], color='magenta', lw=1.5, label='YOLO perpendicular'),
]
ax.legend(handles=[Line2D([0], [0], color='red', lw=2.0, label='Trajectory')] + legend_elements)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Trajectory + oriented YOLO centroids over Offline OSM (with Cluster IDs)")
plt.show()

# -------------- Quick plot in ODOM coordinates (metri) --------------
plt.figure(figsize=(10, 10)) # Dimensione aumentata
plt.plot(odom_x, odom_y, color="red", linewidth=1.5, label="Odometry")

# --- Modifica: Aggiungi ID al plot ODOM ---
if cent_segments_odom:
    for (x_s, y_s, x_e, y_e, lab, c_id) in cent_segments_odom:
        if lab == "parallel":
            color = "limegreen"
        elif lab == "perpendicular":
            color = "magenta"
        else:
            color = "yellow"
        
        # Disegna il dash
        plt.plot([x_s, x_e], [y_s, y_e], color=color, linewidth=1.5)
        
        # Aggiungi il testo ID
        cx = (x_s + x_e) / 2
        cy = (y_s + y_e) / 2
        
        plt.text(
            cx, 
            cy, 
            f"ID:{c_id}", 
            color='black', 
            fontsize=8, 
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5), 
            ha='center', 
            va='center',
            zorder=6
        )

plt.xlabel("X [m] (odom)")
plt.ylabel("Y [m] (odom)")
plt.axis("equal")
plt.grid(True, alpha=0.35)
plt.legend(handles=[
    Line2D([0], [0], color='red', lw=1.5, label='Odometry'),
    Line2D([0], [0], color='limegreen', lw=1.5, label='Centroid parallel'),
    Line2D([0], [0], color='magenta', lw=1.5, label='Centroid perpendicular'),
])
plt.title("Odometry + YOLO centroids as oriented dashes (ODOM frame, with Cluster IDs)")
plt.show()

# -------------------- Steering: target vs feedback --------------------
def plot_steering(cmd_ts, cmd_vals, fb_ts, fb_vals, title_prefix="Steering"):
    if cmd_ts.size == 0 or fb_ts.size == 0:
        print("No steering samples to plot.")
        return
    uniq_cmd_mask = np.concatenate([[True], np.diff(cmd_ts) > 0])
    uniq_fb_mask  = np.concatenate([[True], np.diff(fb_ts) > 0])
    cmd_ts_u, cmd_vals_u = cmd_ts[uniq_cmd_mask], cmd_vals[uniq_cmd_mask]
    fb_ts_u,  fb_vals_u  = fb_ts[uniq_fb_mask],  fb_vals[uniq_fb_mask]
    fb_on_cmd = np.interp(cmd_ts_u, fb_ts_u, fb_vals_u)

    plt.figure()
    plt.plot(cmd_ts_u, cmd_vals_u, linewidth=1.0, label="steering_target")
    plt.plot(cmd_ts_u, fb_on_cmd,  linewidth=1.0, label="steering_pct (interp)")
    plt.xlabel("Time since start [s]")
    plt.ylabel("Steering")
    plt.title(f"{title_prefix}: target vs feedback")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure()
    plt.scatter(cmd_vals_u, fb_on_cmd, s=4)
    plt.xlabel("steering_target")
    plt.ylabel("steering_pct")
    plt.title(f"{title_prefix}: feedback vs target (scatter)")
    plt.grid(True, alpha=0.3)
    plt.show()


cmd_ts  = float_ts.get(cmd_topic,  np.array([]))
cmd_val = float_vals.get(cmd_topic, np.array([]))
fb_ts   = float_ts.get(feedback_topic,  np.array([]))
fb_val  = float_vals.get(feedback_topic, np.array([]))
plot_steering(cmd_ts, cmd_val, fb_ts, fb_val, title_prefix="Steering")

# -------------------- Optional quick sanity prints --------------------
def stats(arr):
    return {
        'count': int(arr.size),
        'mean': float(np.mean(arr)) if arr.size else float('nan'),
        'std': float(np.std(arr)) if arr.size else float('nan'),
        'min': float(np.min(arr)) if arr.size else float('nan'),
        'max': float(np.max(arr)) if arr.size else float('nan'),
    }

print("Yaw stats:", stats(odom_yaw))
print(f"Start lat/lon = ({lat[0]:.8f}, {lon[0]:.8f}); End = ({lat[-1]:.8f}, {lon[-1]:.8f})")

# -------------- ONLINE OSM BASEMAP OVERLAY (via contextily) --------------
try:
    provider = cx.providers.CartoDB.Positron
except Exception:
    provider = cx.providers.OpenStreetMap.Mapnik

fig, ax = plt.subplots(figsize=(10, 10)) # Dimensione aumentata
ax.plot(x_merc, y_merc, linewidth=2.0, label="trajectory", color="red", zorder=4)

# --- Modifica: Aggiungi ID al plot Online Basemap ---
if cent_segments_odom:
    for (x_s, y_s, x_e, y_e, lab, c_id) in cent_segments_odom:
        lat_seg, lon_seg = odom_xy_to_wgs84_vec(
            np.array([x_s, x_e]), np.array([y_s, y_e])
        )
        xs_merc, ys_merc = transformer_to_3857.transform(
            lon_seg.astype(float), lat_seg.astype(float)
        )

        if lab == "parallel":
            color = "limegreen"
        elif lab == "perpendicular":
            color = "magenta"
        else:
            color = "yellow"

        # Disegna il dash
        ax.plot(xs_merc, ys_merc, linewidth=1.5, color=color, alpha=0.9, zorder=5)
        
        # Aggiungi il testo ID
        cx_merc = (xs_merc[0] + xs_merc[1]) / 2
        cy_merc = (ys_merc[0] + ys_merc[1]) / 2

        ax.text(
            cx_merc, 
            cy_merc, 
            f"ID:{c_id}", 
            color='black',
            fontsize=8, 
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5),
            ha='center', 
            va='center',
            zorder=6
        )

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Trajectory + oriented YOLO centroids over Online Basemap (with Cluster IDs)")
cx.add_basemap(ax, crs="EPSG:3857", source=provider)
ax.legend(handles=[
    Line2D([0], [0], color='red', lw=2.0, label='Trajectory'),
    Line2D([0], [0], color='limegreen', lw=1.5, label='Centroid parallel'),
    Line2D([0], [0], color='magenta', lw=1.5, label='Centroid perpendicular'),
])
plt.show()
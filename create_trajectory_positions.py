#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create trajectory_positions.json in maps/<map_name>/
using Lane-Based Heading Calculation (matching dataset generation logic).

Input format (position.txt):
# FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Yaw, Lat, Lon
"""

import os
import json
import math
import argparse
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, MultiLineString

# ==========================================
# 📦 IMPORTS (Must match your project structure)
# ==========================================
from config import MAPS_FOLDER_NAME, SPAWN_OFFSET_METERS
from utils.map_data import (
    fetch_osm_data,
    generate_spawn_gdf,
    get_origin_lat_lon,
    latlon_to_carla,
    get_heading,
)

# ==========================================
# ⚙️ CONSTANTS & CALIBRATION
# ==========================================
LAT0 = 48.17552100
LON0 = 11.59523900
ODOM0_X = 692925.990
ODOM0_Y = 5339070.997
YAW_OFFSET = -0.00000000

def enu_to_latlon(dx_m, dy_m, lat_ref, lon_ref):
    lat_ref_rad = math.radians(lat_ref)
    m_per_deg_lat = (111132.92 - 559.82 * math.cos(2 * lat_ref_rad) + 1.175 * math.cos(4 * lat_ref_rad))
    m_per_deg_lon = (111412.84 * math.cos(lat_ref_rad) - 93.5 * math.cos(3 * lat_ref_rad))
    dlat_deg = dy_m / m_per_deg_lat
    dlon_deg = dx_m / m_per_deg_lon
    return lat_ref + dlat_deg, lon_ref + dlon_deg

def odom_xy_to_wgs84_vec(x_arr: np.ndarray, y_arr: np.ndarray):
    dx = x_arr.astype(float) - ODOM0_X
    dy = y_arr.astype(float) - ODOM0_Y
    c, s = math.cos(YAW_OFFSET), math.sin(YAW_OFFSET)
    dx_e = c * dx - s * dy
    dy_n = s * dx + c * dy
    lat = np.empty_like(dx_e, dtype=float)
    lon = np.empty_like(dy_n, dtype=float)
    for i in range(dx_e.size):
        lat[i], lon[i] = enu_to_latlon(dx_e[i], dy_n[i], LAT0, LON0)
    return lat, lon

# ==========================================
# 📂 DATA LOADING
# ==========================================
def load_positions(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Position file not found: {path}")

    frames, timestamps, oxs, oys = [], [], [], []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4: continue
            try:
                frames.append(int(parts[0]))
                timestamps.append(float(parts[1]))
                oxs.append(float(parts[2]))
                oys.append(float(parts[3]))
                # We ignore Odom Yaw (parts[4]) entirely!
            except ValueError: continue

    return (np.array(frames), np.array(timestamps), np.array(oxs), np.array(oys))

# ==========================================
# 🚀 MAIN LOGIC
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, help="Map folder name")
    parser.add_argument("--positions", required=True, help="Path to position.txt")
    args = parser.parse_args()

    map_folder = os.path.join(MAPS_FOLDER_NAME, args.map)
    
    # 1. Load Trajectory Data
    frames, times, ox, oy = load_positions(args.positions)
    print(f"[INFO] Loaded {len(frames)} frames.")

    # 2. Load Map Data
    print(f"[INFO] Loading Map Data from {args.map}...")
    _, edges, _ = fetch_osm_data(map_folder)
    
    # Get Anchors
    origin_lat, origin_lon = get_origin_lat_lon(edges, "")
    print(f"[DEBUG] Map Origin: Lat {origin_lat}, Lon {origin_lon}")

    # 3. Convert Odom -> Lat/Lon
    traj_lats, traj_lons = odom_xy_to_wgs84_vec(ox, oy)

    # 4. Generate Spawn GDF (Lanes) for Heading Calculation
    print("[INFO] Generating Lane Geometry to calculate headings...")
    spawn_gdf = generate_spawn_gdf(edges, offset=SPAWN_OFFSET_METERS, override=True)
    
    trajectory_data = []

    print("[INFO] Processing frames...")
    for i in range(len(frames)):
        lat_i = float(traj_lats[i])
        lon_i = float(traj_lons[i])
        pt = Point(lon_i, lat_i)
        
        # Default Heading (if no lane found)
        final_heading = 0.0
        
        # --- LOGIC COPIED FROM YOUR DATASET SCRIPT ---
        if not spawn_gdf.empty:
            # Find nearest lane
            dists = spawn_gdf.geometry.distance(pt)
            min_idx = int(dists.idxmin())
            row = spawn_gdf.loc[min_idx]
            lane_geom = row.geometry
            side = row["side"]
            
            # Handle MultiLineString
            if isinstance(lane_geom, MultiLineString):
                lane_geom = max(lane_geom, key=lambda g: g.length)

            if isinstance(lane_geom, LineString):
                # Calculate Heading from Lane Geometry
                coords = list(lane_geom.coords)
                h = get_heading(coords[0][1], coords[0][0], coords[-1][1], coords[-1][0])
                
                # Adjust for side (Left lanes are usually opposed to digitizing direction)
                if side == "left":
                    h = (h + 180.0) % 360.0
                final_heading = h
        
        # Convert Lat/Lon to CARLA World Coordinates
        cx, cy, _ = latlon_to_carla(origin_lat, origin_lon, lat_i, lon_i)

        # Build JSON Entry
        entry = {
            "frame_id": int(frames[i]),
            "timestamp": float(times[i]),
            "transform": {
                "location": {
                    "x": cx,
                    "y": cy,
                    "z": 0.0  # Default Z
                },
                "rotation": {
                    "pitch": 0.0,
                    "yaw": final_heading,  # THIS IS NOW THE CORRECT LANE HEADING
                    "roll": 0.0
                }
            }
        }
        trajectory_data.append(entry)

    # 5. Save Output
    output_path = os.path.join(map_folder, "trajectory_positions.json")
    with open(output_path, "w") as f:
        json.dump(trajectory_data, f, indent=2)

    print(f"\n✅ Successfully saved correct trajectory to: {output_path}")
    if len(trajectory_data) > 0:
        print(f"   Sample Heading Frame 0: {trajectory_data[0]['transform']['rotation']['yaw']:.4f}")

if __name__ == "__main__":
    main()
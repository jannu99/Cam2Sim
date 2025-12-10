#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create trajectory_positions.json in maps/<map_name>/
using KINEMATIC Heading Calculation (Direction of Travel).
This guarantees no 180-degree flips because it follows the car's motion vector.

Input format (position.txt):
# FrameID, Timestamp_Sec, Odom_X, Odom_Y, Odom_Yaw, Lat, Lon
"""

import os
import json
import math
import argparse
import numpy as np


# ==========================================
# 📦 IMPORTS
# ==========================================
from config import MAPS_FOLDER_NAME
from utils.map_data import (
    fetch_osm_data,
    get_origin_lat_lon,
    latlon_to_carla,
)

# ==========================================
# ⚙️ CONSTANTS & CALIBRATION
# ==========================================
LAT0 = 48.17552100
LON0 = 11.59523900
ODOM0_X = 692925.990
ODOM0_Y = 5339070.997

# --- 🎯 CALIBRATED OFFSETS ---
SHIFT_X    = -4.231
SHIFT_Y    = 6.538
YAW_OFFSET = -0.05435897
# -----------------------------

def enu_to_latlon(dx_m, dy_m, lat_ref, lon_ref):
    lat_ref_rad = math.radians(lat_ref)
    m_per_deg_lat = (111132.92 - 559.82 * math.cos(2 * lat_ref_rad) + 1.175 * math.cos(4 * lat_ref_rad))
    m_per_deg_lon = (111412.84 * math.cos(lat_ref_rad) - 93.5 * math.cos(3 * lat_ref_rad))
    dlat_deg = dy_m / m_per_deg_lat
    dlon_deg = dx_m / m_per_deg_lon
    return lat_ref + dlat_deg, lon_ref + dlon_deg

def odom_xy_to_wgs84_vec(x_arr: np.ndarray, y_arr: np.ndarray):
    # 1. Center coordinates
    dx = x_arr.astype(float) - ODOM0_X
    dy = y_arr.astype(float) - ODOM0_Y
    
    # 2. APPLY CALIBRATED SHIFTS (Translation)
    dx = dx + SHIFT_X
    dy = dy + SHIFT_Y

    # 3. Apply Rotation (Yaw)
    c, s = math.cos(YAW_OFFSET), math.sin(YAW_OFFSET)
    dx_e = c * dx - s * dy
    dy_n = s * dx + c * dy
    
    # 4. Convert to Geodetic (Lat/Lon)
    lat = np.empty_like(dx_e, dtype=float)
    lon = np.empty_like(dy_n, dtype=float)
    for i in range(dx_e.size):
        lat[i], lon[i] = enu_to_latlon(dx_e[i], dy_n[i], LAT0, LON0)
    return lat, lon

def calculate_kinematic_heading(current_pos, next_pos):
    """
    Calculates the heading (yaw) between two points (x, y).
    Returns degrees in [0, 360).
    """
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]
    
    # If the car isn't moving, return None to keep previous heading
    if abs(dx) < 0.001 and abs(dy) < 0.001:
        return None

    # Calculate angle in radians
    angle_rad = math.atan2(dy, dx)
    
    # Convert to degrees
    angle_deg = math.degrees(angle_rad)
    
    # Normalize to [0, 360) for CARLA
    return angle_deg % 360.0

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
    
    # 2. Load Map Data (Only needed for coordinate anchor now)
    print(f"[INFO] Loading Map Data for coordinate anchor...")
    _, edges, _ = fetch_osm_data(map_folder)
    origin_lat, origin_lon = get_origin_lat_lon(edges, "")
    
    # 3. Convert Odom -> Lat/Lon -> CARLA (Applying Shifts)
    print(f"[INFO] Converting coordinates with Shift X:{SHIFT_X}, Y:{SHIFT_Y}...")
    traj_lats, traj_lons = odom_xy_to_wgs84_vec(ox, oy)
    
    # Pre-calculate all CARLA positions to make heading look-ahead easier
    carla_positions = []
    for i in range(len(frames)):
        cx, cy, _ = latlon_to_carla(origin_lat, origin_lon, traj_lats[i], traj_lons[i])
        carla_positions.append((cx, cy))
    
    trajectory_data = []

    print("[INFO] Computing Kinematic Headings...")
    
    LOOKAHEAD = 5  # Look 5 frames ahead to smooth out jitter
    last_valid_heading = 0.0

    for i in range(len(frames)):
        cx, cy = carla_positions[i]
        
        # --- KINEMATIC HEADING LOGIC ---
        # Look ahead to find the direction vector
        target_idx = min(i + LOOKAHEAD, len(frames) - 1)
        
        heading_val = None
        
        # If we are at the very end, look backwards
        if target_idx == i and i > 0:
            prev_idx = max(0, i - LOOKAHEAD)
            heading_val = calculate_kinematic_heading(carla_positions[prev_idx], (cx, cy))
        else:
            heading_val = calculate_kinematic_heading((cx, cy), carla_positions[target_idx])
            
        # If car stopped or data undefined, use last known heading
        if heading_val is not None:
            last_valid_heading = heading_val
            
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
                    "yaw": last_valid_heading,  # Calculated from actual movement
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
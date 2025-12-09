#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create / overwrite spawn_positions in maps/<map_name>/vehicle_data.json
from a file of YOLO cluster centroids expressed in odom coordinates.

Supports input format:
cluster_id, x, y, z, count, conf, orientation, side, R-G-B
Example:
1, 692955.845, 5339059.252, 549.017, 274, 0.695, perpendicular, left, 100-52-57

Usage:
    python create_vehicle_data_from_centroids.py \
        --map guerickestrasse_alte_heide_munich_25_11_03 \
        --centroids yolo_centroids_clusters.txt
"""

import os
import json
import math
import argparse

import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
from collections import Counter

from config import MAPS_FOLDER_NAME, SPAWN_OFFSET_METERS, CAR_SPACING
from utils.map_data import (
    fetch_osm_data,
    generate_spawn_gdf,
    get_origin_lat_lon,
    latlon_to_carla,
    get_heading,
)

# =======================
#  ODOM → LAT/LON (calibration constants)
# =======================

LAT0 = 48.17552100
LON0 = 11.59523900
ALT0 = 0.000
ODOM0_X = 692925.990
ODOM0_Y = 5339070.997

# --- 🎯 UPDATED CALIBRATION VALUES ---
SHIFT_X    = -4.231       # East/West Shift
SHIFT_Y    = 6.538        # North/South Shift
YAW_OFFSET = -0.04666667  # Rotation (Radians)
# -------------------------------------


def enu_to_latlon(dx_m, dy_m, lat_ref, lon_ref):
    """
    Convert local ENU offsets (east, north) to WGS84 lat/lon.
    """
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
    """
    Vectorized conversion from odom XY to (lat, lon).
    """
    # 1. Center around ODOM origin
    dx = x_arr.astype(float) - ODOM0_X
    dy = y_arr.astype(float) - ODOM0_Y

    # 2. APPLY CALIBRATED SHIFTS
    dx = dx + SHIFT_X
    dy = dy + SHIFT_Y

    # 3. Apply Rotation
    c, s = math.cos(YAW_OFFSET), math.sin(YAW_OFFSET)
    dx_e = c * dx - s * dy  # east
    dy_n = s * dx + c * dy  # north

    lat = np.empty_like(dx_e, dtype=float)
    lon = np.empty_like(dy_n, dtype=float)
    for i in range(dx_e.size):
        lat[i], lon[i] = enu_to_latlon(dx_e[i], dy_n[i], LAT0, LON0)
    
    return lat, lon


# =======================
#  Centroid loader (UPDATED)
# =======================

def load_centroids_xy(path: str):
    """
    Parses file format:
    cluster_id, x, y, z, count, last_conf, orientation, side, rgb_color
    Example: 1, 692955.845, ..., perpendicular, left, 100-52-57

    Returns:
        cluster_ids, x, y, orientations, sides, colors
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Centroid file not found: {path}")

    ids = []
    xs = []
    ys = []
    orientations = []
    sides = []
    colors = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]
            # We need at least id, x, y (3 columns)
            if len(parts) < 3:
                continue

            # --- ID, X, Y ---
            try:
                cid = int(float(parts[0]))
                x_val = float(parts[1])
                y_val = float(parts[2])
            except ValueError:
                continue

            # --- orientation (index 6) ---
            orient = "unknown"
            if len(parts) >= 7:
                o = parts[6].lower()
                if o in ("parallel", "perpendicular"):
                    orient = o

            # --- side (index 7) ---
            side = "unknown"
            if len(parts) >= 8:
                s = parts[7].lower()
                if s in ("left", "right"):
                    side = s

            # --- COLOR (index 8) ---
            # Format expected: "100-52-57" -> Convert to "100,52,57"
            color_str = None
            if len(parts) >= 9:
                raw_color = parts[8]  # e.g. "100-52-57"
                try:
                    c_parts = raw_color.split('-')
                    if len(c_parts) == 3:
                        r = int(c_parts[0])
                        g = int(c_parts[1])
                        b = int(c_parts[2])
                        color_str = f"{r},{g},{b}"
                except ValueError:
                    pass

            ids.append(cid)
            xs.append(x_val)
            ys.append(y_val)
            orientations.append(orient)
            sides.append(side)
            colors.append(color_str)

    if not xs:
        raise RuntimeError(f"Failed to load centroids from {path}: no valid lines parsed.")

    id_arr = np.array(ids, dtype=int)
    x_arr  = np.array(xs, dtype=float)
    y_arr  = np.array(ys, dtype=float)
    
    return id_arr, x_arr, y_arr, orientations, sides, colors


# =======================
#  Spawn positions builder
# =======================

def build_spawn_positions_from_centroids(cluster_ids, cent_x, cent_y,
                                         orientation, side_labels, colors, edges):
    """
    Maps centroids to parking lines and assigns properties (including color).
    """

    assert cent_x.size == cent_y.size == len(cluster_ids) == len(orientation) == len(side_labels) == len(colors), \
        "All input arrays must have the same length"

    print(f"[INFO] Building spawn_positions for {cent_x.size} centroids...")

    origin_lat, origin_lon = get_origin_lat_lon(edges, "")
    print(f"[INFO] Origin lat/lon from edges centroid: {origin_lat:.8f}, {origin_lon:.8f}")

    cent_lat, cent_lon = odom_xy_to_wgs84_vec(cent_x, cent_y)

    spawn_gdf = generate_spawn_gdf(edges, offset=SPAWN_OFFSET_METERS, override=True)
    if spawn_gdf.empty:
        raise RuntimeError("spawn_gdf is empty: no parking lines generated.")

    spawn_positions = []
    L = CAR_SPACING * 1.1 
    print(f"[INFO] spawn_gdf has {len(spawn_gdf)} parking lines.")

    for i in range(cent_lat.size):
        lat_i = float(cent_lat[i])
        lon_i = float(cent_lon[i])
        pt = Point(lon_i, lat_i)

        side_label = str(side_labels[i]).lower()
        if side_label not in ("left", "right"):
            side_label = "unknown"

        # Filter by side
        if side_label in ("left", "right"):
            cand = spawn_gdf[spawn_gdf["side"] == side_label]
            if cand.empty:
                cand = spawn_gdf
        else:
            cand = spawn_gdf

        # Find nearest line
        dists = cand.geometry.distance(pt)
        min_idx = int(dists.idxmin())
        row = cand.loc[min_idx]
        parking_line = row.geometry
        side = row["side"]

        if isinstance(parking_line, MultiLineString):
            parking_line = max(parking_line, key=lambda g: g.length)

        if not isinstance(parking_line, LineString):
            continue

        # Project
        proj_dist = parking_line.project(pt)
        proj_point = parking_line.interpolate(proj_dist)
        proj_lat, proj_lon = proj_point.y, proj_point.x

        # proj_lat=lat_i
        # proj_lon=lon_i
        # Heading
        coords_pl = list(parking_line.coords)
        start_lat_pl, start_lon_pl = coords_pl[0][1], coords_pl[0][0]
        end_lat_pl,   end_lon_pl   = coords_pl[-1][1], coords_pl[-1][0]
        line_heading = get_heading(start_lat_pl, start_lon_pl, end_lat_pl, end_lon_pl)

        if side == "left":
            line_heading = (line_heading + 180.0) % 360.0

        raw_mode = str(orientation[i]).lower()
        mode = raw_mode if raw_mode in ("parallel", "perpendicular") else "parallel"

        veh_heading = line_heading
        if mode == "perpendicular":
            veh_heading = (veh_heading + 90.0) % 360.0

        line_heading_rad = math.radians(line_heading)

        # To CARLA coords
        start_pos = latlon_to_carla(origin_lat, origin_lon, proj_lat, proj_lon)

        #start_pos = [carla_x, carla_y, 0.0]
        #end_pos   = [
         #   carla_x + 0.0001 ,#L * math.cos(line_heading_rad),
         #   carla_y + 0.0001 #L * math.sin(line_heading_rad),
         #   0.0,
        #]
        increment = (0.0001, 0.0001, 0.0)

        # Calculate end_pos by adding corresponding elements
        end_pos = [s + i for s, i in zip(start_pos, increment)]

        street_id = row.get("osmid_str", row.get("osmid", None))

        spawn_positions.append({
            "cluster_id": int(cluster_ids[i]),
            "side": side,
            "street_id": street_id,
            "mode": mode,
            "start": start_pos,
            "end":   end_pos,
            "heading": veh_heading,
            "color": colors[i]  # "R,G,B"
        })

    print(f"[INFO] Created {len(spawn_positions)} spawn segments.")
    side_mode = Counter((s["side"], s["mode"]) for s in spawn_positions)
    print("[INFO] Spawn side/mode counts:")
    for (side, mode), n in sorted(side_mode.items()):
        print(f"   {side:5s} {mode:12s}: {n}")

    return spawn_positions


# =======================
#  Main
# =======================

def main():
    parser = argparse.ArgumentParser(
        description="Overwrite spawn_positions in vehicle_data.json from odom centroids."
    )
    parser.add_argument(
        "--map", required=True,
        help="Map folder name (without 'maps/')"
    )
    parser.add_argument(
        "--centroids", required=True,
        help="Path to centroid file"
    )
    args = parser.parse_args()

    map_folder = os.path.join(MAPS_FOLDER_NAME, args.map)
    if not os.path.isdir(map_folder):
        raise FileNotFoundError(f"Map folder not found: {map_folder}")

    centroid_file = args.centroids
    
    # LOAD DATA
    cluster_ids, cent_x, cent_y, orientation, side_labels, colors = load_centroids_xy(centroid_file)
    
    if cent_x.size == 0:
        raise RuntimeError(f"No centroids loaded from {centroid_file}.")

    print(f"[INFO] Loaded {cent_x.size} centroids from {centroid_file}")

    _, edges, _ = fetch_osm_data(map_folder)

    # BUILD POSITIONS
    spawn_positions = build_spawn_positions_from_centroids(
        cluster_ids, cent_x, cent_y, orientation, side_labels, colors, edges
    )

    # SAVE
    vehicle_data_path = os.path.join(map_folder, "vehicle_data.json")
    if not os.path.exists(vehicle_data_path):
        raise FileNotFoundError(
            f"vehicle_data.json not found in {map_folder}.\n"
            f"First run create_map.py to generate it."
        )

    with open(vehicle_data_path, "r") as f:
        vehicle_data = json.load(f)

    vehicle_data["spawn_positions"] = spawn_positions

    with open(vehicle_data_path, "w") as f:
        json.dump(vehicle_data, f, indent=2)

    print(f"\n✅ Updated spawn_positions in: {vehicle_data_path}")
    print(f"   Number of parked cars (segments): {len(spawn_positions)}")


if __name__ == "__main__":
    main()
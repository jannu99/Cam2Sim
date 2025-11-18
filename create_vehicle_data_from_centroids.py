#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create / overwrite spawn_positions in maps/<map_name>/vehicle_data.json
from a file of YOLO cluster centroids expressed in odom coordinates.

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
from shapely.geometry import Point

from config import MAPS_FOLDER_NAME, SPAWN_OFFSET_METERS, CAR_SPACING
from utils.map_data import (
    fetch_osm_data,
    generate_spawn_gdf,
    get_origin_lat_lon,
    latlon_to_carla,
    get_heading,
)

# =======================
#  ODOM → LAT/LON (copy of your plotting script logic)
#  Adjust these constants only if you recalibrate
# =======================

LAT0 = 48.17552100
LON0 = 11.59523900
ALT0 = 0.000
ODOM0_X = 692925.990
ODOM0_Y = 5339070.997
YAW_OFFSET = -0.04000000  # radians (CCW positive)


def enu_to_latlon(dx_m, dy_m, lat_ref, lon_ref):
    """
    Convert local ENU offsets (east, north) to WGS84 lat/lon.

    Same approximation you used in the plotting script.
    """
    lat_ref_rad = math.radians(lat_ref)
    # meters per degree (approx)
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
    Vectorized conversion from odom XY to (lat, lon) using the same
    math and constants as your plotting script.
    """
    dx = x_arr.astype(float) - ODOM0_X
    dy = y_arr.astype(float) - ODOM0_Y

    c, s = math.cos(YAW_OFFSET), math.sin(YAW_OFFSET)
    dx_e = c * dx - s * dy  # east
    dy_n = s * dx + c * dy  # north

    lat = np.empty_like(dx_e, dtype=float)
    lon = np.empty_like(dy_n, dtype=float)
    for i in range(dx_e.size):
        lat[i], lon[i] = enu_to_latlon(dx_e[i], dy_n[i], LAT0, LON0)
    return lat, lon


# =======================
#  Centroid loader
# =======================

def load_centroids_xy(path: str):
    import os
    import numpy as np

    if not os.path.exists(path):
        raise FileNotFoundError(f"Centroid file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        arr = np.atleast_2d(arr)
        if arr.shape[1] >= 3 and np.all(np.mod(arr[:, 0], 1) == 0):
            x, y = arr[:, 1].astype(float), arr[:, 2].astype(float)
        else:
            x, y = arr[:, 0].astype(float), arr[:, 1].astype(float)
        return x, y

    # text formats (.txt / .csv)
    try:
        # Load only first 6 numeric columns
        raw = np.loadtxt(path, delimiter=",", comments="#", usecols=(0,1,2,3,4,5))
        if raw.shape[1] >= 3 and np.all(np.mod(raw[:, 0], 1) == 0):
            x, y = raw[:, 1].astype(float), raw[:, 2].astype(float)
        else:
            x, y = raw[:, 0].astype(float), raw[:, 1].astype(float)
        return x, y
    except Exception as e:
        raise RuntimeError(f"Failed to load centroids from {path}: {e}")


# =======================
#  Spawn positions from centroids
# =======================

def build_spawn_positions_from_centroids(cent_x, cent_y, edges):
    """
    - Convert centroids (odom XY) → lat/lon
    - Compute origin_lat/lon from edges
    - Build spawn_gdf with a parking offset
    - For each centroid:
        * find nearest spawn line
        * project centroid onto that line
        * compute heading along the line
        * convert projected point → carla local XY
        * create a short segment around that point so that
          spawn_parked_cars(...) will spawn exactly one car.
    """
    print(f"[INFO] Building spawn_positions for {cent_x.size} centroids...")

    # 1. map origin in WGS84
    origin_lat, origin_lon = get_origin_lat_lon(edges, "")
    print(f"[INFO] Origin lat/lon from edges centroid: {origin_lat:.8f}, {origin_lon:.8f}")

    # 2. build parking lines (override=True -> allow parking on all roads)
    spawn_gdf = generate_spawn_gdf(edges, offset=SPAWN_OFFSET_METERS, override=True)
    if spawn_gdf.empty:
        raise RuntimeError("spawn_gdf is empty: no parking lines generated. "
                           "Check your OSM data / parking tags / edges.")

    # 3. convert centroids odom XY → lat/lon
    cent_lat, cent_lon = odom_xy_to_wgs84_vec(cent_x, cent_y)

    spawn_positions = []
    L = CAR_SPACING * 1.1  # segment length → ensures num_cars = 1

    for i in range(cent_lat.size):
        lat_i = float(cent_lat[i])
        lon_i = float(cent_lon[i])
        pt = Point(lon_i, lat_i)

        # find nearest parking line
        dists = spawn_gdf.geometry.distance(pt)
        min_idx = int(dists.idxmin())
        min_dist = float(dists[min_idx])
        row = spawn_gdf.loc[min_idx]
        line = row.geometry
        side = row["side"]

        # project centroid onto that line
        proj_dist = line.project(pt)
        proj_point = line.interpolate(proj_dist)
        proj_lat, proj_lon = proj_point.y, proj_point.x

        # heading of the parking line at the endpoints
        coords = list(line.coords)
        start_lat, start_lon = coords[0][1], coords[0][0]
        end_lat, end_lon = coords[-1][1], coords[-1][0]
        heading = get_heading(start_lat, start_lon, end_lat, end_lon)
        if side == "left":
            heading = (heading + 180.0) % 360.0

        heading_rad = math.radians(heading)

        # convert projected point to local carla coordinates
        carla_x, carla_y, _ = latlon_to_carla(origin_lat, origin_lon, proj_lat, proj_lon)

        # build a short segment centered on (carla_x, carla_y)
        dx = 0.5 * L * math.cos(heading_rad)
        dy = 0.5 * L * math.sin(heading_rad)

        start_pos = [carla_x - dx, carla_y - dy, 0.0]
        end_pos   = [carla_x + dx, carla_y + dy, 0.0]

        spawn_positions.append({
            "side": side,
            "street_id": None,          # not used by spawn_parked_cars
            "mode": "parallel",         # always parallel parking from centroids
            "start": start_pos,
            "end":   end_pos,
            "heading": heading
        })

    print(f"[INFO] Created {len(spawn_positions)} spawn segments.")
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
        help="Map folder name (without 'maps/'), e.g. guerickestrasse_alte_heide_munich_25_11_03"
    )
    parser.add_argument(
        "--centroids", required=True,
        help="Path to centroid file (e.g. yolo_centroids_clusters.txt)"
    )
    args = parser.parse_args()

    map_folder = os.path.join(MAPS_FOLDER_NAME, args.map)
    if not os.path.isdir(map_folder):
        raise FileNotFoundError(f"Map folder not found: {map_folder}")

    centroid_file = args.centroids
    cent_x, cent_y = load_centroids_xy(centroid_file)
    if cent_x.size == 0:
        raise RuntimeError(f"No centroids loaded from {centroid_file}.")

    print(f"[INFO] Loaded {cent_x.size} centroids from {centroid_file}")

    # load OSM-derived edges
    _, edges, _ = fetch_osm_data(map_folder)

    # compute new spawn_positions
    spawn_positions = build_spawn_positions_from_centroids(cent_x, cent_y, edges)

    # load existing vehicle_data.json (must exist so we keep hero_car + offset)
    vehicle_data_path = os.path.join(map_folder, "vehicle_data.json")
    if not os.path.exists(vehicle_data_path):
        raise FileNotFoundError(
            f"vehicle_data.json not found in {map_folder}.\n"
            f"First run create_map.py (e.g. with --mode all) to generate it."
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

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
YAW_OFFSET = 0  # radians (CCW positive)


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
    """
    Ritorna:
        cluster_ids: np.ndarray shape (N,)
        x, y      : np.ndarray shape (N,)
        orientation: list[str] (parallel|perpendicular|unknown)
        side      : list[str] (left|right|unknown) se presente nel file, altrimenti 'unknown'
    Atteso formato .txt/.csv:
        cluster_id, x, y, z, count, conf, orientation, [side]
    Esempio:
        1, 692934.742, 5339060.430, 549.472, 65, 0.901, parallel, left
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Centroid file not found: {path}")

    ids = []
    xs = []
    ys = []
    orientations = []
    sides = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]
            # voglio almeno: id, x, y
            if len(parts) < 3:
                continue

            # --- ID, X, Y ---
            try:
                cid = int(float(parts[0]))
                x_val = float(parts[1])
                y_val = float(parts[2])
            except ValueError:
                # linea rotta → skip
                continue

            # --- orientation ---
            orient = "unknown"
            if len(parts) >= 7:
                o = parts[6].lower()
                if o in ("parallel", "perpendicular"):
                    orient = o

            # --- side (se presente) ---
            side = "unknown"
            if len(parts) >= 8:
                s = parts[7].lower()
                if s in ("left", "right"):
                    side = s

            ids.append(cid)
            xs.append(x_val)
            ys.append(y_val)
            orientations.append(orient)
            sides.append(side)

    if not xs:
        raise RuntimeError(f"Failed to load centroids from {path}: no valid lines parsed.")

    id_arr = np.array(ids, dtype=int)
    x_arr  = np.array(xs, dtype=float)
    y_arr  = np.array(ys, dtype=float)
    return id_arr, x_arr, y_arr, orientations, sides


# =======================
#  Spawn positions from centroids
# =======================

from shapely.geometry import Point, LineString, MultiLineString
from collections import Counter

def build_spawn_positions_from_centroids(cluster_ids, cent_x, cent_y,
                                         orientation, side_labels, edges):
    """
    Per ogni centroid:
      1) odom (x,y) → (lat, lon)
      2) genera spawn_gdf (linee di parcheggio left/right)
      3) sceglie SOLO le linee del lato richiesto (side_labels[i]) se disponibile
      4) proietta il centroid sulla linea di parcheggio scelta
      5) heading dalla linea + orientation (parallel/perpendicular)
      6) lat/lon proiettato → carla (latlon_to_carla)
      7) crea segmentino (start/end) per spawn_parked_cars

    Nota: side del JSON ora viene dal file, non più stimato dalla strada.
    """

    assert cent_x.size == cent_y.size == len(cluster_ids) == len(orientation) == len(side_labels), \
        "cent_x, cent_y, cluster_ids, orientation, side_labels devono avere stessa lunghezza"

    print(f"[INFO] Building spawn_positions for {cent_x.size} centroids...")

    # 1. origin lat/lon dalla mappa (stesso di create_map.py)
    origin_lat, origin_lon = get_origin_lat_lon(edges, "")
    print(f"[INFO] Origin lat/lon from edges centroid: {origin_lat:.8f}, {origin_lon:.8f}")

    # 2. converti tutti i centroidi da odom → WGS84
    cent_lat, cent_lon = odom_xy_to_wgs84_vec(cent_x, cent_y)

    # 3. genera le linee di parcheggio da edges (sia left che right)
    spawn_gdf = generate_spawn_gdf(edges, offset=SPAWN_OFFSET_METERS, override=True)
    if spawn_gdf.empty:
        raise RuntimeError("spawn_gdf is empty: no parking lines generated. "
                           "Check your OSM data / parking tags / edges.")

    spawn_positions = []
    L = CAR_SPACING * 1.1  # lunghezza segmentino → 1 sola macchina
    print(f"[INFO] spawn_gdf has {len(spawn_gdf)} parking lines.")

    for i in range(cent_lat.size):
        lat_i = float(cent_lat[i])
        lon_i = float(cent_lon[i])
        pt = Point(lon_i, lat_i)

        # side da file (può essere 'left', 'right' o 'unknown')
        side_label = str(side_labels[i]).lower()
        if side_label not in ("left", "right"):
            side_label = "unknown"

        # --- 3a) filtra solo linee dello stesso lato (se definito) ---
        if side_label in ("left", "right"):
            cand = spawn_gdf[spawn_gdf["side"] == side_label]
            if cand.empty:
                # fallback → tutte le linee
                cand = spawn_gdf
        else:
            cand = spawn_gdf

        # --- 3b) distanza sulla gdf candidata ---
        dists = cand.geometry.distance(pt)
        min_idx = int(dists.idxmin())
        row = cand.loc[min_idx]
        parking_line = row.geometry
        side = row["side"]  # questo è il side che andrà nel JSON

        if isinstance(parking_line, MultiLineString):
            # prendi il ramo più lungo
            parking_line = max(parking_line, key=lambda g: g.length)

        if not isinstance(parking_line, LineString):
            continue

        # 4) proietta il centroido su quella linea di parcheggio
        proj_dist = parking_line.project(pt)
        proj_point = parking_line.interpolate(proj_dist)
        proj_lat, proj_lon = proj_point.y, proj_point.x

        # 5) heading della linea di parcheggio
        coords_pl = list(parking_line.coords)
        start_lat_pl, start_lon_pl = coords_pl[0][1], coords_pl[0][0]
        end_lat_pl,   end_lon_pl   = coords_pl[-1][1], coords_pl[-1][0]
        line_heading = get_heading(start_lat_pl, start_lon_pl, end_lat_pl, end_lon_pl)

        # stessa convenzione di prima: flip per left
        if side == "left":
            line_heading = (line_heading + 180.0) % 360.0

        # orientation dal file
        raw_mode = str(orientation[i]).lower()
        if raw_mode not in ("parallel", "perpendicular"):
            mode = "parallel"
        else:
            mode = raw_mode

        # heading veicolo:
        #   parallel      → lungo la linea
        #   perpendicular → linea + 90°
        veh_heading = line_heading
        if mode == "perpendicular":
            veh_heading = (veh_heading + 90.0) % 360.0

        line_heading_rad = math.radians(line_heading)

        # 6) punto proiettato -> coordinate carla locali
        carla_x, carla_y, _ = latlon_to_carla(origin_lat, origin_lon, proj_lat, proj_lon)

        # 7) segmentino: start = posizione auto, end = spostato lungo la linea
        start_pos = [carla_x, carla_y, 0.0]
        end_pos   = [
            carla_x + L * math.cos(line_heading_rad),
            carla_y + L * math.sin(line_heading_rad),
            0.0,
        ]

        # street_id opzionale (per debug)
        street_id = row.get("osmid_str", row.get("osmid", None))

        spawn_positions.append({
            "cluster_id": int(cluster_ids[i]),
            "side": side,          # lato effettivo della linea di parcheggio
            "street_id": street_id,
            "mode": mode,          # "parallel" / "perpendicular"
            "start": start_pos,
            "end":   end_pos,
            "heading": veh_heading # heading finale veicolo
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
    cluster_ids, cent_x, cent_y, orientation, side_labels = load_centroids_xy(centroid_file)
    if cent_x.size == 0:
        raise RuntimeError(f"No centroids loaded from {centroid_file}.")

    print(f"[INFO] Loaded {cent_x.size} centroids from {centroid_file}")

    _, edges, _ = fetch_osm_data(map_folder)

    spawn_positions = build_spawn_positions_from_centroids(
        cluster_ids, cent_x, cent_y, orientation, side_labels, edges
    )

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

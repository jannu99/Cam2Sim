import json
import os.path

from config import MAPS_FOLDER_NAME, SPAWN_OFFSET_METERS, SPAWN_OFFSET_METERS_LEFT, SPAWN_OFFSET_METERS_RIGHT
from utils.argparser import parse_map_args
from utils.map_data import get_street_data, fetch_osm_data, generate_spawn_gdf, get_origin_lat_lon, latlon_to_carla, get_heading
from utils.other import ensure_carla_functionality
from utils.plotting import create_plot, show_plot, get_output
from utils.save_data import get_map_folder_name, create_map_folders, save_vehicle_data, save_map_data, save_osm_data, get_map_data, get_existing_osm_data

from shapely.geometry import LineString


# =======================
# AUTO-GENERATION MODE
# =======================

def generate_all_segments(edges, origin_lat, origin_lon, folder_name, dist_value):

    spawn_gdf = generate_spawn_gdf(edges, offset=SPAWN_OFFSET_METERS,  offset_left=SPAWN_OFFSET_METERS_LEFT,  offset_right=SPAWN_OFFSET_METERS_RIGHT)

    print(f"[AUTO] Trovati {len(spawn_gdf)} segmenti di parcheggio possibili")

    spawn_positions = []

    for idx, row in spawn_gdf.iterrows():
        geom = row.geometry
        side = row["side"]

        if not isinstance(geom, LineString):
            continue

        coords = list(geom.coords)
        start_lon, start_lat = coords[0]
        end_lon, end_lat = coords[-1]

        heading = get_heading(start_lat, start_lon, end_lat, end_lon)
        if side == "left":
            heading = (heading + 180) % 360

        carla_start = latlon_to_carla(origin_lat, origin_lon, start_lat, start_lon)
        carla_end   = latlon_to_carla(origin_lat, origin_lon, end_lat, end_lon)

        spawn_positions.append({
            "side": side,
            "street_id": None,
            "mode": "parallel",
            "start": [carla_start[0], carla_start[1], 0.0],
            "end":   [carla_end[0],   carla_end[1],   0.0],
            "heading": heading
        })

    # Hero car = primo segmento
    hero = spawn_positions[0]

    vehicle_json = {
        "offset": {"x": None, "y": None, "heading": None},
        "dist": dist_value,
        "hero_car": {
            "position": hero["start"],
            "heading": hero["heading"]
        },
        "spawn_positions": spawn_positions
    }

    out_path = os.path.join(folder_name, "vehicle_data.json")
    with open(out_path, "w") as f:
        json.dump(vehicle_json, f, indent=2)

    print(f"\n✅ AUTO-MODE COMPLETATO")
    print(f"📂 File salvato in: {out_path}")


# =======================
# MAIN SCRIPT
# =======================

args = parse_map_args()
map_name = args.name if args.name else get_map_folder_name(args.address)
folder_name = os.path.join(MAPS_FOLDER_NAME, map_name)
map_data = get_map_data(map_name, None, args.no_carla)

if not args.no_carla:
    ensure_carla_functionality()

# FETCH OSM
if not args.skip_fetch:
    osm_data = get_street_data(args.address, dist=args.dist)
else:
    osm_data = get_existing_osm_data(folder_name)

create_map_folders(folder_name)

if not args.skip_fetch:
    save_osm_data(folder_name, osm_data)

G, edges, buildings = fetch_osm_data(folder_name)


# ========================
# NEW: AUTO MODE
# ========================
if args.mode == "all":
    print("\n===============================")
    print("   AUTO-MODE ATTIVO (no GUI)   ")
    print("===============================\n")

    # 1️⃣ salva la mappa OSM come fa il ramo manuale
    save_map_data(folder_name, osm_data, args.no_carla)

    # 2️⃣ calcola origin lat/lon
    origin_lat, origin_lon = get_origin_lat_lon(edges, "")

    # 3️⃣ crea vehicle_data.json automaticamente
    generate_all_segments(edges, origin_lat, origin_lon, folder_name, args.dist)

    print("\n📂 Map + Vehicle data salvati in:", folder_name)
    exit()


# ========================
# MANUAL MODE (GUI)
# ========================

create_plot(buildings, edges, args.address)
show_plot()

output_json = get_output(args.dist)

if map_data is not None and map_data["vehicle_data"]["dist"] == args.dist and map_data["vehicle_data"]["offset"] is not None:
    output_json["offset"] = map_data["vehicle_data"]["offset"]
    print("\n⚠️ Copied Offset Values from existing Map-Data.")

if output_json is not None:
    save_map_data(folder_name, osm_data, args.no_carla)
    save_vehicle_data(folder_name, output_json)
    print("\n📂 Map saved successfully to:", folder_name)
else:
    print("\n⚠️ No Data available to save.")

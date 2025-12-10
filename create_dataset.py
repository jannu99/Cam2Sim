import sys
import os
import json
import math
import argparse
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
from huggingface_hub import login, HfApi
from dotenv import load_dotenv
from datasets import Dataset, Image  # <--- IMPORTED IMAGE HERE

from config import DATASETS_FOLDER_NAME, MAPS_FOLDER_NAME, SPAWN_OFFSET_METERS, SPAWN_OFFSET_METERS_LEFT, SPAWN_OFFSET_METERS_RIGHT
from utils.argparser import parse_dataset_args
from utils.dataset import (
    get_video_dataset, 
    create_segmentation_data, 
    create_canny_data
)
from utils.save_data import (
    create_dataset_folders, 
    get_dataset_folder_name, 
    save_dataset, 
    create_dotenv, 
)
from utils.map_data import (
    fetch_osm_data,
    generate_spawn_gdf,
    get_origin_lat_lon,
    latlon_to_carla,
    get_heading,
)

# ==========================================
# ⚙️ CONSTANTS
# ==========================================
LAT0 = 48.17552100
LON0 = 11.59523900
ODOM0_X = 692925.990
ODOM0_Y = 5339070.997
SHIFT_X    = -4.231
SHIFT_Y    = 6.538
YAW_OFFSET = -0.05435897
# ----------------------------------------

INSTANCE_FOLDER_NAME = "instance"

# ==========================================
# 📐 MATH HELPER FUNCTIONS
# ==========================================

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

def load_odom_frames_xy(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Positions file not found: {path}")
    ids, xs, ys = [], [], []
    with open(path, "r") as f:
        for line in f:
            if line.strip().startswith("#"): continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4: continue
            try:
                cid = int(float(parts[0]))
                x_val = float(parts[2])
                y_val = float(parts[3])
                ids.append(cid)
                xs.append(x_val)
                ys.append(y_val)
            except ValueError:
                continue
    return np.array(ids, dtype=int), np.array(xs, dtype=float), np.array(ys, dtype=float)

# ==========================================
# 🗺️ MAP PROJECTION LOGIC
# ==========================================

def calculate_frame_projections(frame_ids, cent_x, cent_y, edges):
    print(f"[INFO] Processing {cent_x.size} frames (Direct Spawn Mode)...")
    
    cent_lat, cent_lon = odom_xy_to_wgs84_vec(cent_x, cent_y)
    origin_lat, origin_lon = get_origin_lat_lon(edges, "")
    print(f"DEBUG: Map Origin is Lat: {origin_lat}, Lon: {origin_lon}") 

    spawn_gdf = generate_spawn_gdf(edges, offset=SPAWN_OFFSET_METERS, offset_left=SPAWN_OFFSET_METERS_LEFT,  offset_right=SPAWN_OFFSET_METERS_RIGHT, override=True)
    
    results = {}
    
    for i in range(cent_lat.size):
        fid = int(frame_ids[i])
        lat_i = float(cent_lat[i])
        lon_i = float(cent_lon[i])
        pt = Point(lon_i, lat_i)
        
        # 1. Convert RAW GPS to Carla Coordinates immediately
        raw_x, raw_y, _ = latlon_to_carla(origin_lat, origin_lon, lat_i, lon_i)
        
        # Lift slightly to avoid z-fighting with road
        start_pos = (raw_x, raw_y, 0.0)
        end_pos = (raw_x + 0.0001, raw_y + 0.0001, 0.0)
        
        final_heading = 0.0
        street_id = None
        side = "unknown"
        mode = "direct"

        # 2. Use Map ONLY for Heading and Metadata
        if not spawn_gdf.empty:
            dists = spawn_gdf.geometry.distance(pt)
            min_idx = int(dists.idxmin())
            row = spawn_gdf.loc[min_idx]
            lane_geom = row.geometry
            side = row["side"]
            street_id = str(row.get("osmid", ""))
            
            if isinstance(lane_geom, MultiLineString):
                lane_geom = max(lane_geom, key=lambda g: g.length)

            if isinstance(lane_geom, LineString):
                proj_dist = lane_geom.project(pt)
                coords = list(lane_geom.coords)
                h = get_heading(coords[0][1], coords[0][0], coords[-1][1], coords[-1][0])
                if side == "left":
                    h = (h + 180.0) % 360.0
                final_heading = h

        results[fid] = {
            "start": start_pos,
            "end": end_pos,
            "heading": final_heading,
            "street_id": street_id,
            "side": side,
            "mode": mode
        }
    return results

# ==========================================
# 📂 DATASET UTILS
# ==========================================

def process_instance_folder(dataset_path, dataset_list):
    instance_path = os.path.join(dataset_path, INSTANCE_FOLDER_NAME)
    if not os.path.exists(instance_path): return dataset_list
    print(f"🔄 Processing Instance Maps in: {instance_path}...")
    for filename in os.listdir(instance_path):
        if filename.startswith("map_") and filename.endswith(".png"):
            new_name = filename.replace("map_", "frame_")
            os.rename(os.path.join(instance_path, filename), os.path.join(instance_path, new_name))
            
    for item in dataset_list:
        rgb_name = os.path.basename(item['image'])
        inst_path = os.path.join(instance_path, rgb_name)
        if os.path.exists(inst_path):
            item['instance'] = inst_path
        else:
            try:
                fid = int(rgb_name.split('_')[-1].split('.')[0])
                for f in os.listdir(instance_path):
                    if f.startswith("frame_") and f.endswith(".png"):
                        if int(f.split('_')[-1].split('.')[0]) == fid:
                            item['instance'] = os.path.join(instance_path, f)
                            break
            except: item['instance'] = None
    return dataset_list

def add_temporal_pairs(dataset):
    print("⏳ Adding 'previous' column for Temporal Consistency...")
    def get_frame_id(item):
        try: return int(os.path.basename(item['image']).split('_')[-1].split('.')[0])
        except: return 0
    dataset.sort(key=get_frame_id)
    
    count = 0
    for i in range(len(dataset)):
        if i == 0: dataset[i]['previous'] = dataset[i]['image']
        else: dataset[i]['previous'] = dataset[i-1]['image']
        count += 1
    print(f"✅ Added temporal links to {count} frames.")
    return dataset

def apply_calculated_data(dataset, calculation_results, output_folder):
    json_output_path = os.path.join(output_folder, "vehicle_data.json")
    json_data = []
    print("📝 Applying map calculations to dataset...")
    count = 0
    for item in dataset:
        try:
            fname = os.path.basename(item['image'])
            name_part = fname.split('.')[0]
            fid = int(name_part.split('_')[-1]) if "_" in name_part else int(name_part)

            if fid in calculation_results:
                res = calculation_results[fid]
                x, y, z = res['start']
                item['text'] = f"pos x {round(x, 1)}, y {round(y, 1)}"
                json_data.append({
                    "frame_id": fid, "street_id": res['street_id'], "mode": res['mode'], 
                    "side": res['side'], "start": res['start'], "end": res['end'], "heading": res['heading']
                })
                count += 1
            else: item['text'] = "pos unknown"
        except Exception: item['text'] = "pos error"

    with open(json_output_path, 'w') as f:
        json.dump({"spawn_positions": json_data}, f, indent=2)
    print(f"✅ Processed {count} frames.")
    print(f"💾 Extended data saved to: {os.path.abspath(json_output_path)}")
    return dataset

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================

# 1. ARGS & SETUP
# Hack to handle strict argparse
json_only_mode = False
if "--json_only" in sys.argv:
    json_only_mode = True
    sys.argv.remove("--json_only")

args = parse_dataset_args()
args.json_only = json_only_mode

# Inject Map/Frames if missing from strict parser
if "--map" in sys.argv and not hasattr(args, 'map'):
    parser = argparse.ArgumentParser()
    parser.add_argument("--map")
    parser.add_argument("--frames")
    extra, _ = parser.parse_known_args()
    if extra.map: args.map = extra.map
    if extra.frames: args.frames = extra.frames

if not hasattr(args, 'map') or not args.map or not hasattr(args, 'frames') or not args.frames:
    raise ValueError("❌ Missing arguments! Please provide --map <name> and --frames <file.txt>")

create_dotenv()
load_dotenv()
huggingface_token = os.getenv("HF_TOKEN")
if huggingface_token: login(token=huggingface_token)

# 2. PATHS
dataset_name = args.name if args.name else get_dataset_folder_name(args.video)
dataset_path = os.path.join(DATASETS_FOLDER_NAME, dataset_name)
create_dataset_folders(dataset_path)

# ==========================================
# ⚡ FAST PATH: JSON ONLY
# ==========================================
if args.json_only:
    print("⚡ Running in JSON-ONLY mode. Skipping image processing...")
    
    map_folder = os.path.join(MAPS_FOLDER_NAME, args.map)
    print(f"📍 Loading positions from {args.frames}...")
    frame_ids, cent_x, cent_y = load_odom_frames_xy(args.frames)
    
    print(f"🗺️ Loading Map Data from {args.map}...")
    _, edges, _ = fetch_osm_data(map_folder)
    
    calc_results = calculate_frame_projections(frame_ids, cent_x, cent_y, edges)
    
    json_data = []
    print("📝 Formatting JSON data...")
    for fid, res in calc_results.items():
        json_data.append({
            "frame_id": fid, 
            "street_id": res['street_id'], 
            "mode": res['mode'], 
            "side": res['side'], 
            "start": res['start'], 
            "end": res['end'], 
            "heading": res['heading']
        })

    json_output_path = os.path.join(dataset_path, "vehicle_data.json")
    with open(json_output_path, 'w') as f:
        json.dump({"spawn_positions": json_data}, f, indent=2)
    print(f"✅ JSON saved to: {json_output_path}")

    if args.upload and huggingface_token:
        print(f"☁️ Uploading vehicle_data.json to HuggingFace...")
        api = HfApi(token=huggingface_token)
        try:
            api.upload_file(
                path_or_fileobj=json_output_path,
                path_in_repo="vehicle_data.json",
                repo_id=f"{api.whoami()['name']}/{dataset_name}",
                repo_type="dataset"
            )
            print("💻 JSON upload successful!")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
    
    print("👋 JSON Job Complete. Exiting.")
    exit(0)

# ==========================================
# 🐢 SLOW PATH: FULL IMAGE PIPELINE
# ==========================================

dataset = get_video_dataset(dataset_path, args.video, args.canny, args.blur, override=args.override)
print("🎨 Generating Segmentation Masks...")
create_segmentation_data(dataset, dataset_path)
if args.canny: create_canny_data(dataset, dataset_path)

dataset = process_instance_folder(dataset_path, dataset)
dataset = add_temporal_pairs(dataset) 

if args.map and args.frames:
    map_folder = os.path.join(MAPS_FOLDER_NAME, args.map)
    print(f"📍 Loading positions from {args.frames}...")
    frame_ids, cent_x, cent_y = load_odom_frames_xy(args.frames)
    print(f"🗺️ Loading Map Data from {args.map}...")
    _, edges, _ = fetch_osm_data(map_folder)
    
    calc_results = calculate_frame_projections(frame_ids, cent_x, cent_y, edges)
    dataset = apply_calculated_data(dataset, calc_results, dataset_path)

print("📦 Packaging dataset for upload...")
hf_dataset = Dataset.from_list(dataset)

print("🖼️  Casting file paths to Image objects...")
hf_dataset = hf_dataset.cast_column("image", Image())
hf_dataset = hf_dataset.cast_column("segmentation", Image())
hf_dataset = hf_dataset.cast_column("instance", Image())
hf_dataset = hf_dataset.cast_column("previous", Image())

save_dataset(dataset_path, hf_dataset)
print("\n📂 Dataset saved successfully to:", dataset_path)

if args.upload and huggingface_token:
    print(f"\n☁️ Uploading to HuggingFace as: {dataset_name}...")
    hf_dataset.push_to_hub(dataset_name, private=True, token=huggingface_token)
    
    print(f"☁️ Uploading vehicle_data.json...")
    api = HfApi(token=huggingface_token)
    api.upload_file(
        path_or_fileobj=os.path.join(dataset_path, "vehicle_data.json"),
        path_in_repo="vehicle_data.json",
        repo_id=f"{api.whoami()['name']}/{dataset_name}",
        repo_type="dataset"
    )
    print("\n💻 Dataset + JSON privately uploaded to Huggingface successfully!")
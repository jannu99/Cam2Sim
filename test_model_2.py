import carla
import time
import os
import json
import numpy as np
import cv2
from PIL import Image

from dotenv import load_dotenv
from huggingface_hub import login
from config import HERO_VEHICLE_TYPE, CARLA_IP, CARLA_PORT, ROTATION_DEGREES, OUTPUT_FOLDER_NAME
from utils.argparser import parse_testing_args
from utils.carla_simulator import spawn_additional_vehicles, update_synchronous_mode, remap_segmentation_colors, \
    carla_image_to_pil, remove_sensor, delete_all_vehicles, get_spectator_transform, set_spectator_transform, \
    generate_world_map, get_rotation_matrix, get_translation_vector, spawn_parked_cars2, get_transform_position
from utils.pygame_helper import combine_images, setup_pygame, setup_sensor, get_sensor_blueprint, \
    show_image
from utils.save_data import create_dotenv, get_map_data, create_output_folders, get_model_data, save_arguments
from utils.stable_diffusion import load_stable_diffusion_pipeline, generate_image
from utils.distortion import compute_intrinsic_matrix, simulate_distortion_from_pinhole
import torchvision.transforms as transforms
from utils.yolo import calculate_yolo_image, load_yolo_model

# --- ARGS & SETUP ---
args = parse_testing_args()

if not args.only_carla:
    create_dotenv()
    load_dotenv()
    huggingface_token = os.getenv("HF_TOKEN")
    if not huggingface_token:
        raise ValueError("⚠️ Huggingface Token not found. Please set the HF_TOKEN in your .env file.")
    login(token=huggingface_token)

client = carla.Client(args.carla_ip, args.carla_port)
client.set_timeout(40.0) 

tm = client.get_trafficmanager(8000)
world = client.get_world()

model_data = get_model_data(args.model)
if model_data is None:
    print(f"Model '{args.model}' not found.")
    exit(1)

map_data = get_map_data(args.map, (model_data["size"]["x"], model_data["size"]["y"]))
if not map_data:
    print(f"⚠️ Could not find map data.")
    exit(1)

offset_data = map_data["vehicle_data"]["offset"]

# --- GENERATE MAP ---
print("Generating World Map (This increases Actor IDs)...")
world = generate_world_map(client, map_data["xodr_data"])
spectator_transform = get_spectator_transform(world)

blueprint_library = world.get_blueprint_library()
rotation_matrix = get_rotation_matrix(ROTATION_DEGREES)
translation_vector = get_translation_vector(map_data["xodr_data"], offset_data)
starting_location = get_transform_position(map_data["vehicle_data"]["hero_car"]["position"], translation_vector, rotation_matrix)

starting_position = carla.Transform(
    carla.Location(x=starting_location[0], y=starting_location[1], z=starting_location[2]),
    carla.Rotation(pitch=0, yaw=(map_data["vehicle_data"]["hero_car"]["heading"] + ROTATION_DEGREES) % 360, roll=0)
)

# Move Spectator
set_spectator_transform(world, starting_position)

# Sync Mode
update_synchronous_mode(world, tm, True, model_data["camera"]["fps"])

# Spawn Hero
vehicle_bp = blueprint_library.find(HERO_VEHICLE_TYPE)
vehicle = world.spawn_actor(vehicle_bp, starting_position)

# --- STRICT CAR FILTERING ---
vehicle_library = []
all_vehicles = blueprint_library.filter('vehicle.*')
forbidden_keywords = [
    "carlacola", "cybertruck", "t2", "sprinter", "firetruck", "ambulance", "bus", "truck", "van", "bingle", 
    "microlino", "vespa", "yamaha", "kawasaki", "harley", "bh", "gazelle", "diamondback", "crossbike", 
    "century", "omafiets", "low_rider", "ninja", "zx125", "yzf", "fuso", "rosa", "isetta"
]

print("\n🔍 Filtering Vehicles...")
for bp in all_vehicles:
    if bp.has_attribute('number_of_wheels'):
        if int(bp.get_attribute('number_of_wheels')) != 4: continue
    if any(k in bp.id.lower() for k in forbidden_keywords): continue
    vehicle_library.append(bp)

print(f"✅ Selected {len(vehicle_library)} pure car blueprints.")

# --- SPAWN PARKED CARS ---
initial_mapping = spawn_parked_cars2(world, vehicle_library, map_data["vehicle_data"]["spawn_positions"], translation_vector, rotation_matrix)
print(f"Initial Python IDs (Pre-Correction): {list(initial_mapping.keys())}")

# --- SENSORS ---
# Use ORIGINAL SIZE to capture high-res data before distortion/resize
sensor_spawn_point = carla.Transform(carla.Location(x=model_data["camera"]["position"]["x"], y=model_data["camera"]["position"]["y"], z=model_data["camera"]["position"]["z"]), carla.Rotation(pitch=0.0))

# 1. Semantic Segmentation
seg_bp = get_sensor_blueprint(blueprint_library, 'sensor.camera.semantic_segmentation', model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"])
# 2. RGB
rgb_bp = get_sensor_blueprint(blueprint_library, 'sensor.camera.rgb', model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"])
# 3. Depth
depth_bp = get_sensor_blueprint(blueprint_library, 'sensor.camera.depth', model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"])
# 4. Instance Segmentation
inst_bp = get_sensor_blueprint(blueprint_library, 'sensor.camera.instance_segmentation', model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"])
# 5. Optical Flow (NEW)
flow_bp = get_sensor_blueprint(blueprint_library, 'sensor.camera.optical_flow', model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"])

seg_sensor, seg_queue = setup_sensor(seg_bp, sensor_spawn_point, vehicle, world)
rgb_sensor, rgb_queue = setup_sensor(rgb_bp, sensor_spawn_point, vehicle, world)
depth_sensor, depth_queue = setup_sensor(depth_bp, sensor_spawn_point, vehicle, world)
inst_sensor, inst_queue = setup_sensor(inst_bp, sensor_spawn_point, vehicle, world)
flow_sensor, flow_queue = setup_sensor(flow_bp, sensor_spawn_point, vehicle, world) # Setup Flow

vehicle.set_autopilot(True)
pygame_screen, pygame_clock = setup_pygame(model_data["size"]["x"], model_data["size"]["y"], args.only_carla)

# --- SETUP OUTPUT FOLDERS ---
output_dir = None
if not args.no_save:
    output_dir = args.output_dir if args.output_dir else "output_" + str(int(time.time()))
    output_dir = os.path.join(OUTPUT_FOLDER_NAME, output_dir)
    create_output_folders(output_dir)
    save_arguments(args, output_dir)
    
    # Create Flow NPY folder
    flow_dir = os.path.join(output_dir, "flow_npy")
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)

if not args.only_carla:
    pipe = load_stable_diffusion_pipeline(args.model, model_data)
    prev_image = map_data["starting_image"]
    yolo_model = load_yolo_model()

# --- DEFINE TWIN TRANSFORMS ---

# 1. Standard Transform (Bilinear) for RGB and Segmentation (Visuals)
image_transforms = transforms.Compose([
    transforms.Resize(model_data["size"]["x"], interpolation=transforms.InterpolationMode.BILINEAR), 
    transforms.CenterCrop(model_data["size"]["x"])
])

# 2. Instance Transform (Nearest Neighbor) for IDs (Data)
# CRITICAL: Must use NEAREST to prevent creating fake IDs at edges
instance_transforms = transforms.Compose([
    transforms.Resize(model_data["size"]["x"], interpolation=transforms.InterpolationMode.NEAREST), 
    transforms.CenterCrop(model_data["size"]["x"])
])

frame = world.tick()
kill_frame = frame + (args.seconds * model_data["camera"]["fps"])

# --- CALIBRATION FLAG ---
mapping_calibrated = False

try:
     while True:
        frame = world.tick()

        # GET DATA
        seg_data = seg_queue.get()
        rgb_data = rgb_queue.get()
        depth_data = depth_queue.get()
        inst_data = inst_queue.get()
        flow_data = flow_queue.get() # Get Flow

        # --- PROCESS OPTICAL FLOW (FOR SAVING) ---
        # Convert raw buffer to Float32 Numpy Array (Height, Width, 2)
        # This preserves the exact math needed for warping later
        flow_array = np.frombuffer(flow_data.raw_data, dtype=np.float32)
        flow_array = flow_array.reshape((flow_data.height, flow_data.width, 2))

        # Process RGB
        rgb_image_pil = carla_image_to_pil(rgb_data)
        
        # Process Instance (Raw Data for Calibration)
        inst_array = np.frombuffer(inst_data.raw_data, dtype=np.uint8)
        inst_array = inst_array.reshape((inst_data.height, inst_data.width, 4))
        
        # Decode IDs for Calibration: BGRA -> R + G*256 + B*65536
        b = inst_array[:,:,0].astype(np.int32)
        g = inst_array[:,:,1].astype(np.int32)
        r = inst_array[:,:,2].astype(np.int32)
        pixel_ids_map = r + (g << 8) + (b << 16)

        # --- AUTO-CALIBRATION (RUNS ONCE) ---
        if not mapping_calibrated and len(initial_mapping) > 0:
            print("🔍 Calibrating IDs...")
            
            # Sort by size to ignore background (usually largest ID)
            visible_pixel_ids, counts = np.unique(pixel_ids_map, return_counts=True)
            sorted_indices = np.argsort(-counts)
            sorted_ids = visible_pixel_ids[sorted_indices]
            
            candidates = []
            # Skip 0 (Background) and skip the #1 largest object (Likely Road/Sky)
            start_idx = 1 if len(sorted_ids) > 1 else 0
            for uid in sorted_ids[start_idx:]:
                if uid == 0: continue
                candidates.append(uid)
            
            if len(candidates) > 0:
                min_pixel_id = np.min(candidates) 
                min_python_id = min(initial_mapping.keys())
                
                offset = 0
                if min_pixel_id > min_python_id + 1000:
                    offset = min_pixel_id - min_python_id
                
                print(f"   Detected Min Pixel ID (Candidate): {min_pixel_id}")
                print(f"   Calculated Offset: {offset}")

                final_mapping = {}
                for py_id, real_id in initial_mapping.items():
                    render_id = int(py_id + offset) 
                    final_mapping[render_id] = int(real_id)
                
                print(f"✅ Calibrated Mapping keys sample: {list(final_mapping.keys())[:5]}")
                
                if not args.no_save:
                    mapping_path = os.path.join(output_dir, "sim_to_real_mapping.json")
                    with open(mapping_path, "w") as f:
                        json.dump(final_mapping, f, indent=4)
                    print(f"✅ Saved CALIBRATED Mapping to {mapping_path}")
                
                mapping_calibrated = True
            else:
                print("⚠️ No distinct objects visible in first frame. Cannot calibrate yet...")


        # Process Segmentation (Convert Color Space First)
        seg_data.convert(carla.ColorConverter.CityScapesPalette) 
        seg_image = carla_image_to_pil(seg_data)
        seg_image = remap_segmentation_colors(seg_image)
        
        # Prepare Instance PIL (Raw Colors for Transformation)
        instance_vis = np.stack([inst_array[:,:,2], inst_array[:,:,1], inst_array[:,:,0]], axis=2)
        instance_pil = Image.fromarray(instance_vis)

        # --- DISTORTION & TRANSFORM ---
        if "calibration" in model_data["camera"].keys() :
            K_pinhole = compute_intrinsic_matrix(model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"]) 
            K_real = np.array(model_data["camera"]["calibration"]["K"], dtype=np.float32) 
            dist_real = np.array(model_data["camera"]["calibration"]["distortion"], dtype=np.float32)
            
            # Distort Everything (Geometry must match)
            distorted_seg = simulate_distortion_from_pinhole(seg_image, K_pinhole, K_real, dist_real)
            distorted_rgb = simulate_distortion_from_pinhole(rgb_image_pil, K_pinhole, K_real, dist_real)
            # Distortion introduces some interpolation inevitably
            distorted_inst = simulate_distortion_from_pinhole(instance_pil, K_pinhole, K_real, dist_real) 
            
            # Resize & Crop
            final_seg = image_transforms(distorted_seg) 
            final_rgb = image_transforms(distorted_rgb)
            
            # [IMPORTANT] Use NEAREST transform for Instance Map
            final_inst = instance_transforms(distorted_inst)
        else:
            final_seg = image_transforms(seg_image)
            final_rgb = image_transforms(rgb_image_pil)
            final_inst = instance_transforms(instance_pil) # Use NEAREST

        # Process Depth
        depth_data.convert(carla.ColorConverter.LogarithmicDepth) 
        depth_image = carla_image_to_pil(depth_data)
        depth_image_np = np.array(depth_image)
        kernel = np.ones((3,3),np.float32)/10
        depth_image_np = cv2.filter2D(depth_image_np,-1,kernel)
        depth_image = Image.fromarray(depth_image_np)

        # Generation Logic (Optional - Mostly for debugging locally)
        combined = final_rgb
        if not args.only_carla:
            generated_image = generate_image(pipe, final_seg, model_data, prev_image, split = 30, guidance = 3.5, set_seed = True, rotate = False)
            prev_image = generated_image
            _, yolo_image = calculate_yolo_image(yolo_model, generated_image)
            combined = combine_images(final_rgb, yolo_image)

        # Saving
        if not args.no_save:
            final_seg.save(os.path.join(output_dir, "seg", '%06d.png' % frame))
            if not args.only_carla:
                generated_image.save(os.path.join(output_dir, "output", '%06d.png' % frame))
            final_rgb.save(os.path.join(output_dir, "carla", '%06d.png' % frame))
            depth_image.save(os.path.join(output_dir, "depth", '%06d.png' % frame))
            
            # SAVE INSTANCE MAP
            instance_dir = os.path.join(output_dir, "instance")
            if not os.path.exists(instance_dir):
                os.makedirs(instance_dir)
            final_inst.save(os.path.join(instance_dir, '%06d.png' % frame))

            # SAVE OPTICAL FLOW (RAW NPY)
            # We save this so we can perform the warping offline on Colab
            np.save(os.path.join(flow_dir, '%06d.npy' % frame), flow_array)

        show_image(pygame_screen, combined)
        pygame_clock.tick(30)

        if frame >= kill_frame:
            print("Terminating simulation")
            break
            
finally:
    remove_sensor(seg_sensor)
    remove_sensor(depth_sensor)
    remove_sensor(rgb_sensor)
    remove_sensor(inst_sensor)
    remove_sensor(flow_sensor) # Clean up flow
    update_synchronous_mode(world, tm, False)
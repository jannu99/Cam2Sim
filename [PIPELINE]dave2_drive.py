import carla
import time
import numpy as np
import pygame
import os
import sys
import json
import cv2 
import ast
from PIL import Image, ImageOps 
from queue import Queue, Empty
import torchvision.transforms as transforms
from collections import Counter
import hashlib 

# --- FIX: Ensure we run from the script's directory so ./models is found ---
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
print(f"📂 Working Directory set to: {current_dir}")

# Custom Utility Imports
from utils.argparser import parse_testing_args
from utils.carla_simulator import (
    update_synchronous_mode, 
    generate_world_map, 
    get_rotation_matrix, 
    get_translation_vector, 
    get_transform_position,
    remove_sensor, 
    remap_segmentation_colors as remap_seg_colors_util 
)
from utils.pygame_helper import setup_pygame, setup_sensor, get_sensor_blueprint, combine_images, show_image
from utils.save_data import get_model_data, get_map_data, create_output_folders, save_arguments, create_dotenv
from config import HERO_VEHICLE_TYPE, ROTATION_DEGREES, OUTPUT_FOLDER_NAME, MODEL_FOLDER_NAME

# DAVE-2 Specific Imports
from dotenv import load_dotenv
from huggingface_hub import login
from utils.dave2_connection import connect_to_dave2_server, send_image_over_connection
from utils.stable_diffusion import load_stable_diffusion_pipeline, generate_image
from utils.yolo import calculate_yolo_image, load_yolo_model

# ==============================================================================
#  CONFIGURATION (MATCHING SCRIPT A EXACTLY)
# ==============================================================================
CARLA_IP = '127.0.0.1'
CARLA_PORT = 2000
IM_WIDTH = 800
IM_HEIGHT = 503
OUTPUT_FOLDER_NAME = "dataset_output_DAVE2"

# ==============================================================================
#  HELPER FUNCTIONS (MATCHING SCRIPT A LOGIC)
# ==============================================================================

def get_most_frequent_color(pixels):
    if len(pixels) == 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    packed = pixels[:, 0] + (pixels[:, 1] << 8) + (pixels[:, 2] << 16)
    counts = Counter(packed)
    most_common_packed = counts.most_common(1)[0][0]
    r = (most_common_packed) & 0xFF
    g = (most_common_packed >> 8) & 0xFF
    b = (most_common_packed >> 16) & 0xFF
    return np.array([r, g, b], dtype=np.uint8) 

def generate_instance_hash_image(inst_array_3ch):
    b, g, r = inst_array_3ch[:,:,0].astype(np.int32), inst_array_3ch[:,:,1].astype(np.int32), inst_array_3ch[:,:,2].astype(np.int32)
    pixel_ids_map = g + (b << 8)
    r_hash = (pixel_ids_map * 13) % 255
    g_hash = (pixel_ids_map * 37) % 255
    b_hash = (pixel_ids_map * 61) % 255
    hashed_inst = np.dstack((r_hash, g_hash, b_hash)).astype(np.uint8)
    return hashed_inst, pixel_ids_map

def process_instance_map_fixed(inst_data, rgb_image_np):
    # This logic matches your DAVE-2 colorization request, but implemented cleanly
    inst_array = np.frombuffer(inst_data.raw_data, dtype=np.uint8).reshape((inst_data.height, inst_data.width, 4))[:,:,:3] 
    hashed_inst_np, pixel_ids_map = generate_instance_hash_image(inst_array)
    
    unique_colors, counts = np.unique(hashed_inst_np.reshape(-1, 3), axis=0, return_counts=True)
    sorted_indices = np.argsort(-counts)
    predominant_color = unique_colors[sorted_indices[0]] 
    final_colored_inst = np.zeros_like(hashed_inst_np)
    
    for i in sorted_indices:
        color_hash = unique_colors[i]
        count = counts[i]
        is_target_mask = np.all(hashed_inst_np == color_hash, axis=-1)
        
        if np.array_equal(color_hash, predominant_color) or count < 100: 
             final_colored_inst[is_target_mask] = [0, 0, 0] 
        else:
            original_pixels = rgb_image_np[is_target_mask]
            if len(original_pixels) > 0:
                true_color_gt = get_most_frequent_color(original_pixels)
                final_colored_inst[is_target_mask] = true_color_gt
            else:
                final_colored_inst[is_target_mask] = [0, 0, 0]
                 
    return Image.fromarray(final_colored_inst)

def carla_image_to_pil(image) -> Image.Image:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb_array = array[:, :, :3][:, :, ::-1]
    return Image.fromarray(rgb_array)

def remap_segmentation_colors(seg_image):
    arr = np.array(seg_image)
    mask1 = np.all(arr == [0, 0, 0], axis=-1)
    arr[mask1] = [128, 64, 128]
    mask2 = np.all(arr == [70, 130, 180], axis=-1)
    arr[mask2] = [0, 0, 0]
    return Image.fromarray(arr)

def sensor_callback(sensor_data, sensor_queue):
    sensor_queue.put(sensor_data)

def save_data(frame_id, output_dir, final_rgb, final_seg, final_inst, final_depth, generated_image):
    filename = f"{frame_id:06d}"
    final_rgb.save(os.path.join(output_dir, "rgb", f"{filename}.png"))
    final_seg.save(os.path.join(output_dir, "semantic", f"{filename}.png"))
    final_inst.save(os.path.join(output_dir, "instance", f"{filename}.png"))
    final_depth.save(os.path.join(output_dir, "depth", f"{filename}.png"))
    if generated_image:
        generated_image.save(os.path.join(output_dir, "generated", f"{filename}.png"))
    
def cleanup_old_sensors(hero_vehicle):
    print("🧹 Cleaning up old sensors on Hero...")
    world = hero_vehicle.get_world()
    attached = [x for x in world.get_actors() if x.parent and x.parent.id == hero_vehicle.id]
    count = 0
    for sensor in attached:
        if 'sensor' in sensor.type_id:
            sensor.destroy()
            count += 1
    print(f"✅ Removed {count} old sensors.")

# ==============================================================================
#  MAIN LOGIC
# ==============================================================================

def main():
    # --- 0. INITIAL SETUP ---
    args = parse_testing_args()
    create_dotenv()
    load_dotenv()
    huggingface_token = os.getenv("HF_TOKEN")
    if huggingface_token: login(token=huggingface_token)
    
    model_data = get_model_data(args.model)
    # Read FOV from config, but ensure it's a string for Blueprint
    fov_str = str(model_data["camera"]["fov"])
    target_size = model_data["size"]["x"] 

    # --- 1. CONNECT & LOAD MAP ---
    print(f"🔌 Connecting to CARLA at {CARLA_IP}:{CARLA_PORT}...")
    try:
        client = carla.Client(CARLA_IP, CARLA_PORT)
        client.set_timeout(40.0) 
        world = client.get_world()
        tm = client.get_trafficmanager(8000)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    map_data = get_map_data(args.map, (model_data["size"]["x"], model_data["size"]["y"]))
    offset_data = map_data["vehicle_data"]["offset"]
    
    # Load Trajectory for START POSITION calculation only
    traj_path = os.path.join("maps", args.map, "trajectory_positions.json")
    if not os.path.exists(traj_path):
        print(f"❌ Trajectory file not found: {traj_path}")
        return
    with open(traj_path, 'r') as f:
        trajectory_points = json.load(f)
    print(f"📂 Loaded {len(trajectory_points)} points for start pos calculation.")

    # NOTE: NO WORLD GENERATION HERE!
    # world = generate_world_map(client, map_data["xodr_data"]) 
    
    rotation_matrix = get_rotation_matrix(ROTATION_DEGREES)
    translation_vector = get_translation_vector(map_data["xodr_data"], offset_data)
    
    # Calculate Start Position from 1st Trajectory Point
    first_pt = trajectory_points[0]["transform"]
    raw_loc = [first_pt["location"]["x"], first_pt["location"]["y"], first_pt["location"]["z"]]
    
    start_loc_mapped = get_transform_position(raw_loc, translation_vector, rotation_matrix)
    start_yaw_mapped = (first_pt["rotation"]["yaw"] + ROTATION_DEGREES) % 360
    
    start_transform = carla.Transform(
        carla.Location(x=start_loc_mapped[0], y=start_loc_mapped[1], z=start_loc_mapped[2]+1 ),
        carla.Rotation(pitch=0, yaw=start_yaw_mapped, roll=0)
    )

    update_synchronous_mode(world, tm, True, model_data["camera"]["fps"])
    world.tick()

    # --- 4. SPAWN HERO ---
    print("🛠️ Spawning Hero...")
    all_vehicles = world.get_actors().filter('vehicle.*')
    hero_vehicle = None
    for v in all_vehicles:
        if any(x.parent and x.parent.id == v.id for x in world.get_actors()):
            hero_vehicle = v
            break
            
    if hero_vehicle:
        print(f"🚗 Found existing Hero: {hero_vehicle.id}")
        hero_vehicle.set_transform(start_transform)
    else:
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(HERO_VEHICLE_TYPE)
        hero_vehicle = world.spawn_actor(vehicle_bp, start_transform)

    # DAVE-2 NEEDS PHYSICS
    hero_vehicle.set_simulate_physics(True) 
    hero_vehicle.set_autopilot(False)
    
    cleanup_old_sensors(hero_vehicle)

    # --- 5. SENSORS (EXACTLY LIKE SCRIPT A) ---
    cam_config = model_data["camera"]
    # Uses exact logic from Script A: position from config
    sensor_tf = carla.Transform(
        carla.Location(x=cam_config["position"]["x"], y=cam_config["position"]["y"], z=cam_config["position"]["z"]),
        carla.Rotation(pitch=cam_config.get("pitch", 0.0))
    )

    blueprint_library = world.get_blueprint_library()
    
    # EXACT COPY OF SENSOR SETUP FROM SCRIPT A
    rgb_bp = get_sensor_blueprint(blueprint_library, 'sensor.camera.rgb', IM_WIDTH, IM_HEIGHT, fov_str)
    sem_bp = get_sensor_blueprint(blueprint_library, 'sensor.camera.semantic_segmentation', IM_WIDTH, IM_HEIGHT, fov_str)
    inst_bp = get_sensor_blueprint(blueprint_library, 'sensor.camera.instance_segmentation', IM_WIDTH, IM_HEIGHT, fov_str)
    depth_bp = get_sensor_blueprint(blueprint_library, 'sensor.camera.depth', IM_WIDTH, IM_HEIGHT, fov_str)
    flow_bp = get_sensor_blueprint(blueprint_library, 'sensor.camera.optical_flow', IM_WIDTH, IM_HEIGHT, fov_str)
    
    rgb_sensor, rgb_queue = setup_sensor(rgb_bp, sensor_tf, hero_vehicle, world)
    sem_sensor, sem_queue = setup_sensor(sem_bp, sensor_tf, hero_vehicle, world)
    inst_sensor, inst_queue = setup_sensor(inst_bp, sensor_tf, hero_vehicle, world)
    depth_sensor, depth_queue = setup_sensor(depth_bp, sensor_tf, hero_vehicle, world)
    flow_sensor, flow_queue = setup_sensor(flow_bp, sensor_tf, hero_vehicle, world) 

    world.tick()

    # --- 6. MODEL SETUP ---
    pygame_screen, pygame_clock = setup_pygame(target_size, target_size, args.only_carla)
    spectator = world.get_spectator()
    
    pipe = None
    yolo_model = None
    prev_image = map_data["starting_image"]

    if args.only_carla:
        print("ℹ️  Running in ONLY_CARLA mode. Stable Diffusion and YOLO disabled.")
    else:
        check_path = os.path.join(current_dir, MODEL_FOLDER_NAME, args.model)
        if not os.path.exists(check_path):
            print(f"⚠️  WARNING: Model folder not found at: {check_path}")
        
        pipe = load_stable_diffusion_pipeline(args.model, model_data)
        yolo_model = load_yolo_model()
        print("✅ Generative Pipeline Loaded.")

    dave2_conn = connect_to_dave2_server()

    # --- FOLDERS ---
    output_dir = os.path.join(OUTPUT_FOLDER_NAME, args.output_dir if args.output_dir else "dave2_data_" + str(int(time.time())))
    create_output_folders(output_dir) 
    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "semantic"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "instance"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "generated"), exist_ok=True)
    
    save_arguments(args, output_dir)
    
    # --- 7. DAVE-2 CONTROL LOOP ---
    print(f"🚀 Starting DAVE-2 Control Loop...")
    
    frame = world.tick()
    kill_frame = frame + (args.seconds * model_data["camera"]["fps"])

    try:
        while True:
            frame = world.tick()
            spectator.set_transform(hero_vehicle.get_transform()) 

            # A. GET DATA
            try:
                seg_data = sem_queue.get(block=True, timeout=1.0)
                rgb_data = rgb_queue.get(block=True, timeout=1.0)
                depth_data = depth_queue.get(block=True, timeout=1.0)
                inst_data = inst_queue.get(block=True, timeout=1.0)
                flow_data = flow_queue.get(block=True, timeout=1.0)
            except Empty:
                print(f"⚠️ Timeout waiting for sensor data at frame {frame}.")
                continue

            # B. PROCESS SENSOR DATA
            # 1. RGB
            rgb_image_np = np.frombuffer(rgb_data.raw_data, dtype=np.uint8).reshape((rgb_data.height, rgb_data.width, 4))[:,:,:3][:,:,::-1]
            rgb_image_pil = Image.fromarray(rgb_image_np)
            
            # 2. Instance
            instance_pil = process_instance_map_fixed(inst_data, rgb_image_np)

            # 3. Semantic
            seg_data.convert(carla.ColorConverter.CityScapesPalette) 
            seg_image = remap_segmentation_colors(carla_image_to_pil(seg_data))

            # 4. Depth
            depth_data.convert(carla.ColorConverter.LogarithmicDepth) 
            depth_image_np = cv2.filter2D(np.array(carla_image_to_pil(depth_data)),-1,np.ones((3,3),np.float32)/10)
            depth_image_pil = ImageOps.invert(Image.fromarray(depth_image_np))
            
            # C. **SIMPLE RESIZE ONLY** (MATCHING SCRIPT A EXACTLY)
            # No calibration, no pinhole math, just resize to model target.
            final_rgb = rgb_image_pil.resize((target_size, target_size), resample=Image.Resampling.LANCZOS)
            final_seg = seg_image.resize((target_size, target_size), resample=Image.Resampling.NEAREST)
            final_inst = instance_pil.resize((target_size, target_size), resample=Image.Resampling.NEAREST)
            final_depth = depth_image_pil.resize((target_size, target_size), resample=Image.Resampling.BILINEAR)

            # D. PREPARE STEERING IMAGE
            steering_image = None
            generated_image = None 
            
            if args.only_carla:
                # ---------------- ONLY CARLA MODE ----------------
                steering_image = final_rgb
                generated_image = final_rgb 
                display_image = final_rgb
            else:
                # ---------------- GEN-AI MODE ----------------
                generated_image = generate_image(pipe, final_seg, model_data, prev_image, split = 30, guidance = 3.5, set_seed = True, rotate = False)
                prev_image = generated_image
                steering_image = generated_image
                _, yolo_image = calculate_yolo_image(yolo_model, generated_image)
                display_image = combine_images(final_rgb, yolo_image)
            
            # E. SEND TO DAVE-2
            steering, throttle = send_image_over_connection(dave2_conn, steering_image)
            
            print(f"Frame: {frame}, Steering: {steering:.4f}, Throttle: {throttle:.4f}")

            # F. APPLY CONTROL
            ackermann_control = carla.VehicleAckermannControl()
            ackermann_control.speed = float(12 / 3.6) 
            ackermann_control.steer = float(steering * -1.0)
            ackermann_control.steer_speed = 0.0
            ackermann_control.acceleration = 0.0
            ackermann_control.jerk = 0.0
            hero_vehicle.apply_ackermann_control(ackermann_control)

            # G. SAVE DATA
            if not args.no_save:
                save_data(frame, output_dir, final_rgb, final_seg, final_inst, final_depth, generated_image)

            # H. DISPLAY
            show_image(pygame_screen, display_image)
            
            pygame_clock.tick(30)
            if frame >= kill_frame: break
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n🛑 User interrupted.")

    finally:
        print("🧹 Cleaning up...")
        if 'rgb_sensor' in locals() and rgb_sensor: remove_sensor(rgb_sensor)
        if 'sem_sensor' in locals() and sem_sensor: remove_sensor(sem_sensor)
        if 'inst_sensor' in locals() and inst_sensor: remove_sensor(inst_sensor)
        if 'depth_sensor' in locals() and depth_sensor: remove_sensor(depth_sensor)
        if 'flow_sensor' in locals() and flow_sensor: remove_sensor(flow_sensor)
        
        update_synchronous_mode(world, tm, False)
        pygame.quit()
        print("👋 Done.")

if __name__ == '__main__':
    main()
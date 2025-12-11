import carla
import time
import numpy as np
import pygame
import os
import sys
import json
import cv2 
import torch
import ast  # <--- Added for color map parsing
from PIL import Image, ImageOps 
from queue import Queue, Empty
import torchvision.transforms as transforms
from collections import Counter
from huggingface_hub import login, snapshot_download
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from safetensors import safe_open

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
from utils.pygame_helper import setup_sensor, get_sensor_blueprint, combine_images, show_image
from utils.save_data import get_model_data, get_map_data, create_output_folders, save_arguments, create_dotenv
from config import HERO_VEHICLE_TYPE, ROTATION_DEGREES, OUTPUT_FOLDER_NAME, MODEL_FOLDER_NAME

# DAVE-2 Specific Imports
from dotenv import load_dotenv
from utils.dave2_connection import connect_to_dave2_server, send_image_over_connection
from utils.yolo import calculate_yolo_image, load_yolo_model

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
CARLA_IP = '127.0.0.1'
CARLA_PORT = 2000
IM_WIDTH = 800
IM_HEIGHT = 503
OUTPUT_FOLDER_NAME = "dataset_output_DAVE2"
INSTANCE_MAP_FILE = "instance_map.txt"  # <--- Added

# --- BEST MODEL PARAMETERS (FROM COLAB CELL 5) ---
STABLE_DIFF_STEPS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scales: Seg 0.7, Inst 0.7, Temp 1.1
COND_SCALES = [0.7, 0.7, 1.1] 

# Schedules (Start/End steps for each controlnet)
# Order: [Segmentation, Instance, Temporal]
CONTROL_START = [0.41, 0.0, 0.0]
CONTROL_END   = [1.0, 0.4, 0.4]

# Static Prompt from your script
STATIC_PROMPT = "street, car, sidewalk, trees, person, trees, cars, trees, cars, trees, trees, trees, trees, trees, trees, trees, trees, trees, trees, trees, trees, trees, trees, trees, trees"
NEGATIVE_PROMPT = "blurry, distorted, low quality, bad anatomy"

# ==============================================================================
#  HELPER FUNCTIONS (CARLA)
# ==============================================================================

def load_instance_color_map(filepath):
    mapped_colors = {}
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: {filepath} not found. No color remapping will occur.")
        return mapped_colors

    try:
        with open(filepath, 'r') as f:
            raw_data = json.load(f)
        for key_str, val_str in raw_data.items():
            rgb_key = ast.literal_eval(key_str)
            rgb_val = [int(x) for x in val_str.split(',')]
            mapped_colors[rgb_key] = rgb_val
        print(f"✅ Loaded {len(mapped_colors)} color mappings.")
    except Exception as e:
        print(f"❌ Error loading color map: {e}")
    return mapped_colors

# Initialize Global Color Map
GLOBAL_COLOR_MAP = load_instance_color_map(INSTANCE_MAP_FILE)

def process_instance_map_fixed(inst_raw_data):
    """
    Applies the specific math hash and color mapping from your reference script.
    """
    bgra = np.frombuffer(inst_raw_data.raw_data, dtype=np.uint8)
    bgra = bgra.reshape((inst_raw_data.height, inst_raw_data.width, 4))
    b_id = bgra[:, :, 0].astype(np.int32) 
    g_id = bgra[:, :, 1].astype(np.int32)
    tag  = bgra[:, :, 2] 
    
    # Filter for vehicles (Tags: 14, 15, 16, 18)
    vehicle_mask = np.isin(tag, [14, 15, 16, 18])
    output = np.zeros((inst_raw_data.height, inst_raw_data.width, 3), dtype=np.uint8)
    
    # Apply Hash Formula
    out_r = (g_id * 37 + b_id * 13) % 200 + 55
    out_g = (g_id * 17 + b_id * 43) % 200 + 55
    out_b = (g_id * 29 + b_id * 53) % 200 + 55
    
    # Apply colors only to masked vehicles
    output[vehicle_mask, 0] = out_r[vehicle_mask]
    output[vehicle_mask, 1] = out_g[vehicle_mask]
    output[vehicle_mask, 2] = out_b[vehicle_mask]
    
    # Apply Global Remapping if it exists
    if GLOBAL_COLOR_MAP:
        for old_rgb, new_rgb in GLOBAL_COLOR_MAP.items():
            source = np.array(old_rgb, dtype=np.uint8)
            target = np.array(new_rgb, dtype=np.uint8)
            mask = np.all(output == source, axis=-1)
            if np.any(mask):
                output[mask] = target

    return Image.fromarray(output)

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
#  HELPER FUNCTIONS (OPTIMIZED GENERATION)
# ==============================================================================

def check_essential_files(model_dir: str) -> bool:
    if not os.path.exists(model_dir): return False
    required = ["config.json", "stable_diffusion/pytorch_lora_weights.safetensors", "controlnet_instance/diffusion_pytorch_model.safetensors"]
    for r in required:
        if not os.path.exists(os.path.join(model_dir, r)): return False
    return True

def download_models_and_config(repo_id, local_dir, token):
    if check_essential_files(local_dir):
        print(f"✅ Models found locally in {local_dir}.")
        return local_dir
    print(f"⏳ Downloading models from {repo_id}...")
    login(token=token, add_to_git_credential=False)
    # Using snapshot_download logic from your script
    return snapshot_download(
        repo_id=repo_id, 
        repo_type="model", 
        local_dir=local_dir, 
        local_dir_use_symlinks=False,
        allow_patterns=[
            "config.json", "lora_weights/*", "stable_diffusion/*", 
            "controlnet_segmentation/*", "controlnet_instance/*", "controlnet_tempconsistency/*"
        ]
    )

def load_pipeline_models(model_root, device):
    config_path = os.path.join(model_root, "config.json")
    with open(config_path, "r") as f:
        model_data = json.load(f)

    print("\n⏳ Loading ControlNet Models...")
    # Load ControlNets
    cnet_seg = ControlNetModel.from_pretrained(os.path.join(model_root, model_data["controlnet_segmentation"]), torch_dtype=torch.float16)
    cnet_temp = ControlNetModel.from_pretrained(os.path.join(model_root, model_data["controlnet_tempconsistency"]), torch_dtype=torch.float16)
    cnet_inst = ControlNetModel.from_pretrained(os.path.join(model_root, model_data["controlnet_instance"]), torch_dtype=torch.float16)

    # Load Pipeline [Seg, Inst, Temp]
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_data["stable_diffusion_model"],
        controlnet=[cnet_seg, cnet_inst, cnet_temp], 
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)

    # Load LoRA
    lora_path = os.path.join(model_root, model_data["lora_weights"])
    print(f"⏳ Loading LoRA from: {lora_path}")
    if lora_path.endswith(".safetensors"):
        lora_state_dict = {}
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys(): lora_state_dict[key] = f.get_tensor(key)
    else:
        lora_state_dict = torch.load(lora_path, map_location="cpu")
    
    pipe.load_lora_weights(lora_state_dict)
    return pipe, model_data

def generate_image_realtime(
    pipe, seg_image, inst_image, model_data, prev_image, guidance=3.0
):
    """
    Generates one frame using the EXACT parameters from your Colab Cell 5.
    """
    
    # 1. Prepare Control Images
    # Temp uses previous image, or falls back to seg_image if None
    ctrl_temp = prev_image if prev_image is not None else seg_image
    
    # ControlNet Input Order: [Seg, Inst, Temp] -> Matches pipeline load order
    control_images = [seg_image, inst_image, ctrl_temp]
    
    # 2. Parameter Configuration (Optimized)
    # Scales: [0.7, 0.7, 1.1] (0.0 for temp on first frame)
    current_temp_scale = 1.1 if prev_image is not None else 0.0
    controlnet_scales = [0.7, 0.7, current_temp_scale]

    generator = torch.Generator(device=pipe.device).manual_seed(50) 

    # 3. Call Pipeline
    with torch.no_grad():
        result = pipe(
            prompt=STATIC_PROMPT,
            image=control_images,
            negative_prompt=NEGATIVE_PROMPT,
            controlnet_conditioning_scale=controlnet_scales,
            height=model_data["size"]["y"],
            width=model_data["size"]["x"],
            num_inference_steps=STABLE_DIFF_STEPS,
            
            # --- APPLIED SCHEDULES ---
            control_guidance_start=CONTROL_START, # [0.41, 0.0, 0.0]
            control_guidance_end=CONTROL_END,     # [1.0, 0.4, 0.4]
            
            guidance_scale=guidance,
            guess_mode=True, # Explicitly TRUE per your script
            output_type="pil",
            generator=generator
        )
    return result.images[0]

# ==============================================================================
#  MAIN LOGIC
# ==============================================================================

def main():
    # --- 0. INITIAL SETUP ---
    args = parse_testing_args()
    create_dotenv()
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    model_data_basic = get_model_data(args.model)
    fov = float(model_data_basic["camera"]["fov"])
    
    # --- FIX: Ensure we have a clean integer size for the grid ---
    target_size = int(model_data_basic["size"]["x"]) 

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

    map_data = get_map_data(args.map, (model_data_basic["size"]["x"], model_data_basic["size"]["y"]))
    offset_data = map_data["vehicle_data"]["offset"]
    
    # Load Trajectory for START POSITION
    traj_path = os.path.join("maps", args.map, "trajectory_positions.json")
    if not os.path.exists(traj_path):
        print(f"❌ Trajectory file not found: {traj_path}")
        return
    with open(traj_path, 'r') as f:
        trajectory_points = json.load(f)
    print(f"📂 Loaded {len(trajectory_points)} points for start pos calculation.")

    rotation_matrix = get_rotation_matrix(ROTATION_DEGREES)
    translation_vector = get_translation_vector(map_data["xodr_data"], offset_data)
    
    first_pt = trajectory_points[0]["transform"]
    raw_loc = [first_pt["location"]["x"], first_pt["location"]["y"], first_pt["location"]["z"]]
    start_loc_mapped = get_transform_position(raw_loc, translation_vector, rotation_matrix)
    start_yaw_mapped = (first_pt["rotation"]["yaw"] + ROTATION_DEGREES) % 360
    
    start_transform = carla.Transform(
        carla.Location(x=start_loc_mapped[0], y=start_loc_mapped[1], z=start_loc_mapped[2]+1 ),
        carla.Rotation(pitch=0, yaw=start_yaw_mapped, roll=0)
    )

    update_synchronous_mode(world, tm, True, model_data_basic["camera"]["fps"])
    world.tick()

    # --- 2. SPAWN HERO ---
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

    hero_vehicle.set_simulate_physics(True) 
    hero_vehicle.set_autopilot(False)
    cleanup_old_sensors(hero_vehicle)

    # --- 3. SENSORS (FIXED 800x503) ---
    cam_config = model_data_basic["camera"]
    sensor_tf = carla.Transform(
        carla.Location(x=cam_config["position"]["x"], y=cam_config["position"]["y"], z=cam_config["position"]["z"]),
        carla.Rotation(pitch=cam_config.get("pitch", 0.0))
    )

    fov_str = str(fov)
    blueprint_library = world.get_blueprint_library()
    
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

    # --- 4. MODEL LOADING (GEN-AI) ---
    # --- FIX: BYPASS 'setup_pygame' entirely. Manually set mode to avoid auto-scaling ---
    win_w = target_size * 2
    win_h = target_size * 2
    print(f"🖥️  Initializing Manual Pygame Window: {win_w} x {win_h} (Square Grid)")

    # REPLACED SETUP:
    pygame.init()
    pygame.display.set_caption("DAVE-2 Gen-AI | 2x2 Grid")
    pygame_screen = pygame.display.set_mode((win_w, win_h))
    pygame_clock = pygame.time.Clock()

    spectator = world.get_spectator()
    
    pipe = None
    yolo_model = None
    prev_image = None # For Temporal Consistency
    model_data_gen = None

    if args.only_carla:
        print("ℹ️  Running in ONLY_CARLA mode. Stable Diffusion and YOLO disabled.")
    else:
        # 1. Download/Cache Models (Using repo from Colab script)
        MODEL_REPO_ID = "jannu99/gurickestraemp4_25_12_02_model" 
        LOCAL_MODEL_DIR = os.path.join(current_dir, "gurickestraemp4_25_12_02_local") 
        
        model_root = download_models_and_config(MODEL_REPO_ID, LOCAL_MODEL_DIR, hf_token)
        
        # 2. Load Pipeline
        pipe, model_data_gen = load_pipeline_models(model_root, DEVICE)
        
        # 3. Load YOLO
        yolo_model = load_yolo_model()
        print("✅ Generative Pipeline & YOLO Loaded.")

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
    
    # --- 5. DRIVING LOOP ---
    print(f"🚀 Starting DAVE-2 Control Loop...")
    
    frame = world.tick()
    kill_frame = frame + (args.seconds * model_data_basic["camera"]["fps"])

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

            # B. PROCESS SENSOR DATA (PIL)
            # 1. RGB
            rgb_image_np = np.frombuffer(rgb_data.raw_data, dtype=np.uint8).reshape((rgb_data.height, rgb_data.width, 4))[:,:,:3][:,:,::-1]
            rgb_image_pil = Image.fromarray(rgb_image_np)
            
            # 2. Instance (UPDATED TO NEW LOGIC)
            instance_pil = process_instance_map_fixed(inst_data)

            # 3. Semantic
            seg_data.convert(carla.ColorConverter.CityScapesPalette) 
            seg_image = remap_segmentation_colors(carla_image_to_pil(seg_data))

            # 4. Depth
            depth_data.convert(carla.ColorConverter.LogarithmicDepth) 
            depth_image_np = cv2.filter2D(np.array(carla_image_to_pil(depth_data)),-1,np.ones((3,3),np.float32)/10)
            depth_image_pil = ImageOps.invert(Image.fromarray(depth_image_np))
            
            # C. **SIMPLE RESIZE ONLY**
            final_rgb = rgb_image_pil.resize((target_size, target_size), resample=Image.Resampling.LANCZOS)
            final_seg = seg_image.resize((target_size, target_size), resample=Image.Resampling.NEAREST)
            final_inst = instance_pil.resize((target_size, target_size), resample=Image.Resampling.NEAREST)
            final_depth = depth_image_pil.resize((target_size, target_size), resample=Image.Resampling.BILINEAR)

            # D. PREPARE IMAGE FOR DAVE-2
            steering_image = None
            generated_image = None 
            image_to_show_top_right = None
            
            if args.only_carla:
                # ---------------- ONLY CARLA MODE ----------------
                steering_image = final_rgb
                generated_image = final_rgb 
                image_to_show_top_right = final_rgb # Top Right shows RGB
            else:
                # ---------------- GEN-AI MODE ----------------
                # Uses the REALTIME function with OPTIMIZED parameters
                generated_image = generate_image_realtime(
                    pipe, 
                    seg_image=final_seg, 
                    inst_image=final_inst, 
                    model_data=model_data_gen, 
                    prev_image=prev_image,
                    guidance=3.0
                )
                
                prev_image = generated_image # Update temporal buffer
                steering_image = generated_image
                
                _, yolo_image = calculate_yolo_image(yolo_model, generated_image)
                image_to_show_top_right = yolo_image # Top Right shows Gen+YOLO
            
            # E. DAVE-2 INFERENCE
            steering, throttle = send_image_over_connection(dave2_conn, steering_image)
            print(f"Frame: {frame}, Steer: {steering:.4f}, Throt: {throttle:.4f}")

            # --- [UPDATED] F. DISPLAY (2x2 GRID) ---
            # 1. Create a blank canvas (Square: Width x 2, Height x 2)
            display_image = Image.new('RGB', (win_w, win_h))

            # 2. Paste images into the quadrants
            # Top-Left: Original RGB
            display_image.paste(final_rgb, (0, 0))
            
            # Top-Right: Generated (with YOLO boxes) OR RGB (if only_carla)
            display_image.paste(image_to_show_top_right, (target_size, 0))
            
            # Bottom-Left: Semantic Segmentation
            display_image.paste(final_seg, (0, target_size))
            
            # Bottom-Right: Instance Segmentation
            display_image.paste(final_inst, (target_size, target_size))

            # G. CONTROL
            ackermann_control = carla.VehicleAckermannControl()
            ackermann_control.speed = float(12 / 3.6) 
            ackermann_control.steer = float(steering * -1.0)
            ackermann_control.steer_speed = 0.0
            ackermann_control.acceleration = 0.0
            ackermann_control.jerk = 0.0
            hero_vehicle.apply_ackermann_control(ackermann_control)

            # H. SAVE
            if not args.no_save:
                save_data(frame, output_dir, final_rgb, final_seg, final_inst, final_depth, generated_image)

            # I. DISPLAY ON SCREEN
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
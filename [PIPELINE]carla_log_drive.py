import carla
import time
import numpy as np
import pygame
import os
import json
import cv2 
import ast
from PIL import Image
from queue import Queue, Empty
import torchvision.transforms as transforms

# Custom Utility Imports
from utils.argparser import parse_testing_args
from utils.carla_simulator import (
    update_synchronous_mode, 
    generate_world_map, 
    get_rotation_matrix, 
    get_translation_vector, 
    get_transform_position
)
from utils.pygame_helper import setup_pygame, setup_sensor, get_sensor_blueprint
from utils.save_data import get_model_data, get_map_data, create_output_folders, save_arguments
from for_transforms import get_inverse_transform
from config import HERO_VEHICLE_TYPE, ROTATION_DEGREES, OUTPUT_FOLDER_NAME

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
CARLA_IP = '127.0.0.1'
CARLA_PORT = 2000
IM_WIDTH = 800
IM_HEIGHT = 503
OUTPUT_FOLDER = "dataset_output"
INSTANCE_MAP_FILE = "instance_map.txt" 

# Create directories
os.makedirs(os.path.join(OUTPUT_FOLDER, "rgb"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "semantic"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "instance"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "data"), exist_ok=True)

# Global Storage
ALL_FRAME_DATA = []

# ==============================================================================
#  HELPER FUNCTIONS
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

GLOBAL_COLOR_MAP = load_instance_color_map(INSTANCE_MAP_FILE)

def remap_segmentation_colors(seg_image):
    arr = np.array(seg_image)
    # Road -> Road Line
    mask1 = np.all(arr == [0, 0, 0], axis=-1)
    arr[mask1] = [128, 64, 128]
    # Sky -> Unlabeled
    mask2 = np.all(arr == [70, 130, 180], axis=-1)
    arr[mask2] = [0, 0, 0]
    return Image.fromarray(arr)

def process_instance_map_fixed(inst_raw_data):
    bgra = np.frombuffer(inst_raw_data.raw_data, dtype=np.uint8)
    bgra = bgra.reshape((inst_raw_data.height, inst_raw_data.width, 4))
    b_id = bgra[:, :, 0].astype(np.int32) 
    g_id = bgra[:, :, 1].astype(np.int32)
    tag  = bgra[:, :, 2] 
    
    vehicle_mask = np.isin(tag, [14, 15, 16, 18])
    output = np.zeros((inst_raw_data.height, inst_raw_data.width, 3), dtype=np.uint8)
    
    out_r = (g_id * 37 + b_id * 13) % 200 + 55
    out_g = (g_id * 17 + b_id * 43) % 200 + 55
    out_b = (g_id * 29 + b_id * 53) % 200 + 55
    
    output[vehicle_mask, 0] = out_r[vehicle_mask]
    output[vehicle_mask, 1] = out_g[vehicle_mask]
    output[vehicle_mask, 2] = out_b[vehicle_mask]
    
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

def sensor_callback(sensor_data, sensor_queue):
    sensor_queue.put(sensor_data)

def save_data(frame_id, rgb_obj, sem_obj, inst_obj, transform_data, img_tf, mask_tf):
    global ALL_FRAME_DATA 
    filename = f"{frame_id:06d}"
    
    # RGB
    rgb_arr = np.frombuffer(rgb_obj.raw_data, dtype=np.uint8)
    rgb_arr = np.reshape(rgb_arr, (rgb_obj.height, rgb_obj.width, 4))[:, :, :3][:, :, ::-1]
    final_rgb = img_tf(Image.fromarray(rgb_arr))
    final_rgb.save(f"{OUTPUT_FOLDER}/rgb/{filename}.png")

    # Semantic
    sem_obj.convert(carla.ColorConverter.CityScapesPalette)
    final_sem = mask_tf(remap_segmentation_colors(carla_image_to_pil(sem_obj)))
    final_sem.save(f"{OUTPUT_FOLDER}/semantic/{filename}.png")

    # Instance
    final_inst = mask_tf(process_instance_map_fixed(inst_obj))
    final_inst.save(f"{OUTPUT_FOLDER}/instance/{filename}.png")

    # Metadata
    ALL_FRAME_DATA.append({
        "frame": frame_id,
        "location": transform_data["location"],
        "rotation": transform_data["rotation"],
        "caption": f"pos x {transform_data['location']['x']:.2f}, y {transform_data['location']['y']:.2f}"
    })

# ==============================================================================
#  MAIN LOGIC
# ==============================================================================

def main():
    args = parse_testing_args()
    model_data = get_model_data(args.model)
    fov = str(model_data["camera"]["fov"])
    target_size = model_data["size"]["x"]

    # Image Transforms
    image_transforms = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(target_size)
    ])
    mask_transforms = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(target_size)
    ])

    print(f"🔌 Connecting to CARLA at {CARLA_IP}:{CARLA_PORT}...")
    try:
        client = carla.Client(CARLA_IP, CARLA_PORT)
        client.set_timeout(20.0)
        world = client.get_world()
        tm = client.get_trafficmanager(8000)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # --- 1. LOAD MAP DATA & TRAJECTORY ---
    map_data = get_map_data(args.map, (model_data["size"]["x"], model_data["size"]["y"]))
    offset_data = map_data["vehicle_data"]["offset"]
    
    # Load Trajectory JSON
    traj_path = os.path.join("maps", args.map, "trajectory_positions.json")
    if not os.path.exists(traj_path):
        print(f"❌ Trajectory file not found: {traj_path}")
        return
        
    with open(traj_path, 'r') as f:
        trajectory_points = json.load(f)
    print(f"📂 Loaded {len(trajectory_points)} trajectory points.")

    # --- 2. CALCULATE TRANSFORMATION MATRICES ---
    # Based on the user provided snippet
    rotation_matrix = get_rotation_matrix(ROTATION_DEGREES)
    translation_vector = get_translation_vector(map_data["xodr_data"], offset_data)

    print("Generating World Map...")
    # Typically we assume map is already loaded in CARLA, but if you need to generate:
    # world = generate_world_map(client, map_data["xodr_data"]) 

    # --- 3. SYNC MODE ---
    update_synchronous_mode(world, tm, True, model_data["camera"]["fps"])
    world.tick()

    # --- 4. SPAWN HERO AT START ---
    print("🔍 Setting up Hero Vehicle...")
    
    # Calculate Start Position from First Trajectory Point
    first_pt = trajectory_points[0]["transform"]
    raw_loc = [first_pt["location"]["x"], first_pt["location"]["y"], first_pt["location"]["z"]]
    
    start_loc_mapped = get_transform_position(raw_loc, translation_vector, rotation_matrix)
    start_yaw_mapped = (first_pt["rotation"]["yaw"] + ROTATION_DEGREES) % 360
    
    # Important: Add z offset +0.5 to prevent floor clipping
    start_transform = carla.Transform(
        carla.Location(x=start_loc_mapped[0], y=start_loc_mapped[1], z=start_loc_mapped[2] + 0.5),
        carla.Rotation(pitch=0, yaw=start_yaw_mapped, roll=0)
    )

    bp_lib = world.get_blueprint_library()
    
    # Try to find existing or spawn new
    all_vehicles = world.get_actors().filter('vehicle.*')
    hero_vehicle = None
    for v in all_vehicles:
        # Check if it has sensors attached (heuristic for hero)
        if any(x.parent and x.parent.id == v.id for x in world.get_actors()):
            hero_vehicle = v
            break
            
    if hero_vehicle:
        print(f"🚗 Found existing Hero: {hero_vehicle.id}")
        hero_vehicle.set_transform(start_transform)
    else:
        print("🛠️ Spawning new Hero...")
        vehicle_bp = bp_lib.find(HERO_VEHICLE_TYPE)
        hero_vehicle = world.spawn_actor(vehicle_bp, start_transform)

    # Disable Physics for Teleportation accuracy
    hero_vehicle.set_simulate_physics(False)
    hero_vehicle.set_autopilot(False)
    print(f"🚗 Hero Configured. Physics Disabled.")

    # --- 5. SENSORS ---
    cleanup_old_sensors(hero_vehicle)
    
    cam_config = model_data["camera"]
    sensor_tf = carla.Transform(
        carla.Location(x=cam_config["position"]["x"], y=cam_config["position"]["y"], z=cam_config["position"]["z"]),
        carla.Rotation(pitch=cam_config.get("pitch", 0.0))
    )

    # Queues
    rgb_queue = Queue()
    sem_queue = Queue()
    inst_queue = Queue()

    # Spawn Sensors
    rgb_bp = bp_lib.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', str(IM_WIDTH))
    rgb_bp.set_attribute('image_size_y', str(IM_HEIGHT))
    rgb_bp.set_attribute('fov', fov)
    rgb_sensor = world.spawn_actor(rgb_bp, sensor_tf, attach_to=hero_vehicle)
    rgb_sensor.listen(lambda d: sensor_callback(d, rgb_queue))

    sem_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    sem_bp.set_attribute('image_size_x', str(IM_WIDTH))
    sem_bp.set_attribute('image_size_y', str(IM_HEIGHT))
    sem_bp.set_attribute('fov', fov)
    sem_sensor = world.spawn_actor(sem_bp, sensor_tf, attach_to=hero_vehicle)
    sem_sensor.listen(lambda d: sensor_callback(d, sem_queue))

    inst_bp = bp_lib.find('sensor.camera.instance_segmentation')
    inst_bp.set_attribute('image_size_x', str(IM_WIDTH))
    inst_bp.set_attribute('image_size_y', str(IM_HEIGHT))
    inst_bp.set_attribute('fov', fov)
    inst_sensor = world.spawn_actor(inst_bp, sensor_tf, attach_to=hero_vehicle)
    inst_sensor.listen(lambda d: sensor_callback(d, inst_queue))

    # --- 6. DISPLAY ---
    pygame.init()
    display = pygame.display.set_mode((IM_WIDTH, IM_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Trajectory Replay")
    clock = pygame.time.Clock()

    print(f"🚀 Starting Trajectory Replay ({len(trajectory_points)} frames)...")

    try:
        for idx, point in enumerate(trajectory_points):
            
            # A. Prepare Transform
            raw_loc = [
                point["transform"]["location"]["x"],
                point["transform"]["location"]["y"],
                point["transform"]["location"]["z"]
            ]
            raw_yaw = point["transform"]["rotation"]["yaw"]

            # Apply Logic: Rotate Vector -> Translate -> Rotate Yaw
            final_loc = get_transform_position(raw_loc, translation_vector, rotation_matrix)
            final_yaw = (raw_yaw + ROTATION_DEGREES) % 360

            # Create CARLA Transform (Z + 0.5 safe offset)
            target_tf = carla.Transform(
                carla.Location(x=final_loc[0], y=final_loc[1], z=final_loc[2] + 0.5),
                carla.Rotation(pitch=0.0, yaw=final_yaw, roll=0.0)
            )

            # B. Teleport
            hero_vehicle.set_transform(target_tf)
            
            # C. Tick World
            world.tick()

            # D. Retrieve Sensor Data
            try:
                rgb_data = rgb_queue.get(block=True, timeout=2.0)
                sem_data = sem_queue.get(block=True, timeout=2.0)
                inst_data = inst_queue.get(block=True, timeout=2.0)
            except Empty:
                print(f"⚠️ Timeout waiting for sensors at frame {idx}")
                continue

            # E. Save Data
            # Note: We pass the *actual* transform from CARLA to save logic for consistency
            actual_tf = hero_vehicle.get_transform()
            current_transform_mapped = get_inverse_transform(actual_tf, args.model, args.map)
            
            save_data(point["frame_id"], rgb_data, sem_data, inst_data, current_transform_mapped, image_transforms, mask_transforms)

            # F. Render RGB to Pygame
            array = np.frombuffer(rgb_data.raw_data, dtype=np.uint8)
            array = np.reshape(array, (rgb_data.height, rgb_data.width, 4))
            array = array[:, :, :3][:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))
            pygame.display.flip()
            pygame.display.set_caption(f"Replay Frame {idx}/{len(trajectory_points)}")

            # G. Inputs
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    raise KeyboardInterrupt
            
            clock.tick(60)

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")

    finally:
        print("💾 Saving metadata...")
        with open(os.path.join(OUTPUT_FOLDER, "data", "all_frame_data.json"), 'w') as f:
            json.dump(ALL_FRAME_DATA, f, indent=4)
        
        print("🧹 Cleaning up...")
        if 'rgb_sensor' in locals() and rgb_sensor: rgb_sensor.destroy()
        if 'sem_sensor' in locals() and sem_sensor: sem_sensor.destroy()
        if 'inst_sensor' in locals() and inst_sensor: inst_sensor.destroy()
        
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        pygame.quit()
        print("👋 Done.")

if __name__ == '__main__':
    main()
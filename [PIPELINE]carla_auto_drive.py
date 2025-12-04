import carla
import time
import numpy as np
import pygame
import os
import json
import cv2 
import ast # <--- REQUIRED TO PARSE TUPLE KEYS
from PIL import Image
from queue import Queue, Empty
from utils.argparser import parse_testing_args
from for_transforms import get_inverse_transform
import torchvision.transforms as transforms # <--- ADD THIS
from utils.save_data import get_model_data  # <--- MAKE SURE THIS IS IMPORTED
# --- CONFIGURATION ---
CARLA_IP = '127.0.0.1'
CARLA_PORT = 2000
IM_WIDTH = 800
IM_HEIGHT = 503
OUTPUT_FOLDER = "dataset_output"
INSTANCE_MAP_FILE = "instance_map.txt" # <--- YOUR MAP FILE

# Create directories
os.makedirs(os.path.join(OUTPUT_FOLDER, "rgb"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "semantic"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "instance"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "data"), exist_ok=True)

# ==============================================================================
#  GLOBAL DATA STORAGE (NEW)
# ==============================================================================
ALL_FRAME_DATA = []

# ==============================================================================
#  COLOR MAPPING LOADER
# ==============================================================================
def load_instance_color_map(filepath):
    """
    Loads the text file: {"(62, 92, 254)": "100,52,57", ...}
    Returns a dict: { (62, 92, 254): [100, 52, 57], ... }
    """
    mapped_colors = {}
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: {filepath} not found. No color remapping will occur.")
        return mapped_colors

    try:
        with open(filepath, 'r') as f:
            raw_data = json.load(f)
            
        for key_str, val_str in raw_data.items():
            # Convert key string "(62, 92, 254)" -> tuple (62, 92, 254)
            rgb_key = ast.literal_eval(key_str)
            
            # Convert value string "100,52,57" -> list [100, 52, 57]
            rgb_val = [int(x) for x in val_str.split(',')]
            
            mapped_colors[rgb_key] = rgb_val
            
        print(f"✅ Loaded {len(mapped_colors)} color mappings from {filepath}")
    except Exception as e:
        print(f"❌ Error loading color map: {e}")
        
    return mapped_colors

# Load this once globally so we don't re-read the file every frame
GLOBAL_COLOR_MAP = load_instance_color_map(INSTANCE_MAP_FILE)

# ==============================================================================
#  HELPER FUNCTIONS
# ==============================================================================
def remap_segmentation_colors(seg_image):
    arr = np.array(seg_image)
    # 0,0,0 → 128,64,128 (Road → Road Line)
    mask1 = np.all(arr == [0, 0, 0], axis=-1)
    arr[mask1] = [128, 64, 128]
    # 70,130,180 → 0,0,0 (Sky → Unlabeled/Other)
    mask2 = np.all(arr == [70, 130, 180], axis=-1)
    arr[mask2] = [0, 0, 0]
    return Image.fromarray(arr)

def process_instance_map_fixed(inst_raw_data):
    """
    1. Generates standard hash colors from CARLA ID.
    2. Overwrites specific colors based on GLOBAL_COLOR_MAP.
    """
    # 1. Convert Raw Buffer to Numpy (BGRA order)
    bgra = np.frombuffer(inst_raw_data.raw_data, dtype=np.uint8)
    bgra = bgra.reshape((inst_raw_data.height, inst_raw_data.width, 4))
    
    # 2. Extract Channels
    b_id = bgra[:, :, 0].astype(np.int32) 
    g_id = bgra[:, :, 1].astype(np.int32)
    tag  = bgra[:, :, 2] # Red Channel is the Semantic Tag
    
    # 3. Create Mask for VEHICLES
    vehicle_mask = np.isin(tag, [14, 15, 16, 18])
    
    # 4. Initialize Output (Black Background)
    output = np.zeros((inst_raw_data.height, inst_raw_data.width, 3), dtype=np.uint8)
    
    # 5. Generate Standard Colors (Hash logic)
    out_r = (g_id * 37 + b_id * 13) % 200 + 55
    out_g = (g_id * 17 + b_id * 43) % 200 + 55
    out_b = (g_id * 29 + b_id * 53) % 200 + 55
    
    # 6. Apply Standard Colors ONLY to Vehicles
    output[vehicle_mask, 0] = out_r[vehicle_mask]
    output[vehicle_mask, 1] = out_g[vehicle_mask]
    output[vehicle_mask, 2] = out_b[vehicle_mask]
    
    # ---------------------------------------------------------
    # 7. APPLY RE-MAPPING FROM HASH MAP (The New Logic)
    # ---------------------------------------------------------
    if GLOBAL_COLOR_MAP:
        # Iterate through the map: Key = Old Color, Value = New Color
        for old_rgb_tuple, new_rgb_list in GLOBAL_COLOR_MAP.items():
            # Convert tuple to numpy array for comparison
            source_color = np.array(old_rgb_tuple, dtype=np.uint8)
            target_color = np.array(new_rgb_list, dtype=np.uint8)
            
            # Find all pixels that match the source color
            # axis=-1 means check R,G,B all match
            mask = np.all(output == source_color, axis=-1)
            
            # If any pixels match, replace them
            if np.any(mask):
                output[mask] = target_color

    return Image.fromarray(output)

# ==============================================================================
#  SENSOR/DATA HANDLING
# ==============================================================================

def cleanup_old_sensors(hero_vehicle):
    print("🧹 Cleaning up old sensors on Hero...")
    world = hero_vehicle.get_world()
    actor_list = world.get_actors()
    attached_sensors = [x for x in actor_list if x.parent and x.parent.id == hero_vehicle.id]
    
    count = 0
    for sensor in attached_sensors:
        if 'sensor' in sensor.type_id:
            sensor.destroy()
            count += 1
    print(f"✅ Removed {count} old sensors.")

def sensor_callback(sensor_data, sensor_queue):
    sensor_queue.put(sensor_data)

def carla_image_to_pil(image) -> Image.Image:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    rgb_array = array[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB
    return Image.fromarray(rgb_array)

def save_data(frame_id, rgb_obj, sem_obj, inst_obj, transform_data, img_tf, mask_tf):
    """
    Saves the image data from the current tick to disk after RESIZING/CROPPING.
    """
    global ALL_FRAME_DATA 
    
    filename = f"{frame_id:06d}"
    
    # --- 1. RGB Processing ---
    # Convert raw data to PIL
    rgb_array = np.frombuffer(rgb_obj.raw_data, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (rgb_obj.height, rgb_obj.width, 4))
    rgb_array = rgb_array[:, :, :3][:, :, ::-1] # BGRA -> RGB
    rgb_pil = Image.fromarray(rgb_array)
    
    # APPLY TRANSFORM (Resize -> 512)
    final_rgb = img_tf(rgb_pil)
    final_rgb.save(f"{OUTPUT_FOLDER}/rgb/{filename}.png")

    # --- 2. Semantic Segmentation Processing ---
    sem_obj.convert(carla.ColorConverter.CityScapesPalette)
    sem_pil_raw = remap_segmentation_colors(carla_image_to_pil(sem_obj))
    
    # APPLY TRANSFORM (Resize -> 512, Nearest Neighbor)
    final_sem = mask_tf(sem_pil_raw)
    final_sem.save(f"{OUTPUT_FOLDER}/semantic/{filename}.png")

    # --- 3. Instance Segmentation Processing ---
    # Get the remapped/processed PIL image first
    inst_pil_raw = process_instance_map_fixed(inst_obj)

    # APPLY TRANSFORM (Resize -> 512, Nearest Neighbor)
    final_inst = mask_tf(inst_pil_raw)
    final_inst.save(f"{OUTPUT_FOLDER}/instance/{filename}.png")

    # --- 4. Save Coordinates (unchanged) ---
    data = {
        "frame": frame_id,
        "location": {
            "x": transform_data["location"]["x"],
            "y": transform_data["location"]["y"],
            "z": transform_data["location"]["z"],
        },
        "rotation": {
            "pitch": transform_data["rotation"]["pitch"],
            "yaw": transform_data["rotation"]["yaw"],
            "roll": transform_data["rotation"]["roll"]
        },
        "caption": f"pos x {transform_data['location']['x']:.2f}, y {transform_data['location']['y']:.2f}",
    }
    ALL_FRAME_DATA.append(data)
# ==============================================================================
#  MAIN LOGIC
# ==============================================================================

def main():
    args = parse_testing_args()
    model=args.model
    map_name=args.map
    # Load model configuration to get target size (e.g., 512)
    model_data = get_model_data(args.model)
    fov = str(model_data["camera"]["fov"]) # <--- ADD THIS LINE HERE
    target_size = model_data["size"]["x"] # e.g. 512

    # 1. Transform for RGB (Smooth interpolation)
    image_transforms = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(target_size)
    ])

    # 2. Transform for MASKS (Semantic/Instance) - MUST use Nearest to preserve colors
    mask_transforms = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(target_size)
    ])
    print(f"🔌 Connecting to CARLA at {CARLA_IP}:{CARLA_PORT}...")
    try:
        client = carla.Client(CARLA_IP, CARLA_PORT)
        client.set_timeout(10.0)
        world = client.get_world()
        tm = client.get_trafficmanager(8000)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # SYNC MODE HANDSHAKE
    settings = world.get_settings()
    if not settings.synchronous_mode:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
    
    print("🕐 Sending initial Tick to wake up server...")
    world.tick()

    # FIND OR SPAWN HERO
    print("🔍 Searching for the Hero Vehicle...")
    all_vehicles = world.get_actors().filter('vehicle.*')
    hero_vehicle = None

    for v in all_vehicles:
        attached = [x for x in world.get_actors() if x.parent and x.parent.id == v.id]
        if len(attached) > 0:
            hero_vehicle = v
            break
    
    if hero_vehicle is None and len(all_vehicles) > 0:
        hero_vehicle = all_vehicles[0]

    if hero_vehicle is None:
        print("🛠️ SPAWNING A NEW HERO...")
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        valid_point = spawn_points[0] if spawn_points else carla.Transform()
        hero_vehicle = world.spawn_actor(vehicle_bp, valid_point)

    print(f"🚗 Hero Vehicle Locked: {hero_vehicle.type_id} (ID: {hero_vehicle.id})")

    # --- SENSOR SETUP ---
    cleanup_old_sensors(hero_vehicle)
    bp_lib = world.get_blueprint_library()
    
    # Common transform
    cam_config = model_data["camera"]
    
    # Use 0.0 for pitch if it's not in the config, matching Script 1's behavior
    cam_pitch = cam_config.get("pitch", 0.0) 
    
    sensor_transform = carla.Transform(
        carla.Location(
            x=cam_config["position"]["x"],
            y=cam_config["position"]["y"],
            z=cam_config["position"]["z"]
        ),
        carla.Rotation(pitch=cam_pitch)
    )
    print(f"📷 Sensor Transform set to: X={cam_config['position']['x']}, Z={cam_config['position']['z']}, Pitch={cam_pitch}")

    # 1. RGB
    rgb_bp = bp_lib.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', str(IM_WIDTH))
    rgb_bp.set_attribute('image_size_y', str(IM_HEIGHT))
    rgb_bp.set_attribute('fov', fov) # <--- PASTE THIS (Replaces '90')
    rgb_sensor = world.spawn_actor(rgb_bp, sensor_transform, attach_to=hero_vehicle)
    rgb_queue = Queue()
    rgb_sensor.listen(lambda data: sensor_callback(data, rgb_queue))

    # 2. Semantic Segmentation
    sem_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    sem_bp.set_attribute('image_size_x', str(IM_WIDTH))
    sem_bp.set_attribute('image_size_y', str(IM_HEIGHT))
    sem_bp.set_attribute('fov', fov) # <--- PASTE THIS (Replaces '90')
    sem_sensor = world.spawn_actor(sem_bp, sensor_transform, attach_to=hero_vehicle)
    sem_queue = Queue()
    sem_sensor.listen(lambda data: sensor_callback(data, sem_queue))

    # 3. Instance Segmentation
    inst_bp = bp_lib.find('sensor.camera.instance_segmentation')
    inst_bp.set_attribute('image_size_x', str(IM_WIDTH))
    inst_bp.set_attribute('image_size_y', str(IM_HEIGHT))
    inst_bp.set_attribute('fov', fov) # <--- PASTE THIS (Replaces '90')
    inst_sensor = world.spawn_actor(inst_bp, sensor_transform, attach_to=hero_vehicle)
    inst_queue = Queue()
    inst_sensor.listen(lambda data: sensor_callback(data, inst_queue))

    # --- AUTOPILOT ---
    print("🤖 Engaging Autopilot...")
    hero_vehicle.set_autopilot(True, tm.get_port())
    tm.ignore_lights_percentage(hero_vehicle, 0)
    tm.distance_to_leading_vehicle(hero_vehicle, 2.5)

    # --- DISPLAY ---
    pygame.init()
    display = pygame.display.set_mode((IM_WIDTH, IM_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption(f"Recording Data")
    clock = pygame.time.Clock()

    print(f"🔴 Recording started. Saving to: {OUTPUT_FOLDER}/")
    print("🏁 Press Q or ESC to stop.")

    frame_count = 0

    try:
        while True:
            # 1. Tick the world
            world.tick()

            # 2. Get data
            try:
                rgb_data = rgb_queue.get(block=True, timeout=2.0)
                sem_data = sem_queue.get(block=True, timeout=2.0)
                inst_data = inst_queue.get(block=True, timeout=2.0)
                current_transform = hero_vehicle.get_transform()
            except Empty:
                continue

            # 3. Save Data (Using fixed colors)
            current_transform_mapped = get_inverse_transform(current_transform, model, map_name)

            # UPDATED CALL: Pass 'image_transforms' and 'mask_transforms'
            save_data(frame_count, rgb_data, sem_data, inst_data, current_transform_mapped, image_transforms, mask_transforms)
            
            # 4. Render RGB
            array = np.frombuffer(rgb_data.raw_data, dtype=np.uint8)
            array = np.reshape(array, (rgb_data.height, rgb_data.width, 4))
            array = array[:, :, :3][:, :, ::-1] # BGRA -> RGB
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))
            pygame.display.flip()

            # 5. Update UI
            pygame.display.set_caption(f"Recording - Frame {frame_count}")
            frame_count += 1
            
            # 6. Inputs
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
            
            clock.tick(60)

    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        print("🧹 Cleaning up sensors...")
        if 'rgb_sensor' in locals() and rgb_sensor: rgb_sensor.destroy()
        if 'sem_sensor' in locals() and sem_sensor: sem_sensor.destroy()
        if 'inst_sensor' in locals() and inst_sensor: inst_sensor.destroy()
        
        # --- NEW: WRITE THE FULL DATA FILE HERE ---
        data_output_path = os.path.join(OUTPUT_FOLDER, "data", "all_frame_data.json")
        print(f"✍️ Saving all {frame_count} frame transforms to {data_output_path}...")
        try:
            with open(data_output_path, 'w') as f:
                json.dump(ALL_FRAME_DATA, f, indent=4)
            print("✅ Transform data saved successfully.")
        except Exception as e:
            print(f"❌ Error saving all_frame_data.json: {e}")
        # ------------------------------------------

        settings.synchronous_mode = False
        world.apply_settings(settings)
        pygame.quit()
        print(f"👋 Done. Saved {frame_count} frames.")

if __name__ == '__main__':
    main()
import carla
import time
import os
import json
import numpy as np
import cv2
import random
import gc 
import pygame 
from PIL import Image, ImageOps 
import torchvision.transforms as transforms
from dotenv import load_dotenv
from huggingface_hub import login
from config import HERO_VEHICLE_TYPE, ROTATION_DEGREES, OUTPUT_FOLDER_NAME
from utils.argparser import parse_testing_args
from utils.carla_simulator import update_synchronous_mode, \
    generate_world_map, get_rotation_matrix, \
    get_translation_vector, get_transform_position
from utils.pygame_helper import setup_pygame, setup_sensor, get_sensor_blueprint, \
    show_image
from utils.save_data import create_dotenv, get_map_data, create_output_folders, get_model_data, save_arguments

# ==============================================================================
#  HELPER FUNCTIONS
# ==============================================================================

def is_spot_occupied(world, target_location, ignore_actor_id, threshold=2.5):
    """
    Checks if a target location is already occupied by another vehicle.
    Calculates Euclidean distance to all other vehicles.
    """
    vehicles = world.get_actors().filter('vehicle.*')
    for vehicle in vehicles:
        if vehicle.id == ignore_actor_id:
            continue 
        
        dist = vehicle.get_location().distance(target_location)
        if dist < threshold:
            return True
    return False

def process_instance_map_fixed(inst_raw_data):
    """Parses CARLA Instance Bitmap for Visualization."""
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
    
    return Image.fromarray(output)

def get_unique_colors_from_sensor(inst_raw_data):
    """Extracts unique RGB colors from the instance sensor data."""
    bgra = np.frombuffer(inst_raw_data.raw_data, dtype=np.uint8)
    bgra = bgra.reshape((inst_raw_data.height, inst_raw_data.width, 4))
    
    b_id = bgra[:, :, 0].astype(np.int32) 
    g_id = bgra[:, :, 1].astype(np.int32)
    tag  = bgra[:, :, 2] 
    
    vehicle_mask = np.isin(tag, [14, 15, 16, 18])
    
    valid_b = b_id[vehicle_mask]
    valid_g = g_id[vehicle_mask]
    
    if len(valid_b) == 0:
        return set()

    out_r = (valid_g * 37 + valid_b * 13) % 200 + 55
    out_g = (valid_g * 17 + valid_b * 43) % 200 + 55
    out_b = (valid_g * 29 + valid_b * 53) % 200 + 55
    
    colors = np.stack((out_r, out_g, out_b), axis=-1)
    unique_colors = np.unique(colors, axis=0)
    
    return set(map(tuple, unique_colors))

def spawn_parked_cars_front_of_hero(world, vehicle_library, entry, hero_position, translation_vector, rotation_matrix):
    spawned_actors = [] 
    
    # 1. Extract color from JSON
    target_color = None
    if "color" in entry and entry["color"]:
        try:
            target_color = [int(x) for x in entry["color"].split(',')]
        except:
            pass
    if target_color is None: target_color = [128, 128, 128]

    # 2. Define temp position (Front of Hero)
    temp_start=[15.420957288852122, -170.21387689825545, 0.0]
    
    start = get_transform_position(temp_start, translation_vector, rotation_matrix)
    heading = (entry["heading"] + ROTATION_DEGREES) % 360

    # 3. Choose Blueprint and Set Color
    blueprint = random.choice(vehicle_library)
    if blueprint.has_attribute('color'):
        color_str = f"{target_color[0]},{target_color[1]},{target_color[2]}"
        blueprint.set_attribute('color', color_str)

    x, y, z = start[0], start[1], start[2] if len(start) > 2 else 0.0

    veh_heading = heading
    if entry["mode"].strip() == "perpendicular" and random.random() < 0.5:
        veh_heading = (veh_heading + 180) % 360

    transform = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=veh_heading))

    try:
        actor = world.spawn_actor(blueprint, transform)
        actor.set_simulate_physics(False)
        spawned_actors.append(actor)
    except RuntimeError:
        pass
    
    return spawned_actors

def flush_all_queues(seg_q, rgb_q, depth_q, inst_q, flow_q):
    """Helper to aggressively empty queues to prevent memory buildup"""
    while not seg_q.empty(): seg_q.get()
    while not rgb_q.empty(): rgb_q.get()
    while not depth_q.empty(): depth_q.get()
    while not inst_q.empty(): inst_q.get()
    while not flow_q.empty(): flow_q.get()

# ==============================================================================
#  MAIN SCRIPT
# ==============================================================================

args = parse_testing_args()

client = carla.Client(args.carla_ip, args.carla_port)
client.set_timeout(40.0) 
tm = client.get_trafficmanager(8000)
world = client.get_world()

model_data = get_model_data(args.model)
map_data = get_map_data(args.map, (model_data["size"]["x"], model_data["size"]["y"]))
offset_data = map_data["vehicle_data"]["offset"]

print("Generating World Map...")
world = generate_world_map(client, map_data["xodr_data"])
blueprint_library = world.get_blueprint_library()
rotation_matrix = get_rotation_matrix(ROTATION_DEGREES)
translation_vector = get_translation_vector(map_data["xodr_data"], offset_data)
starting_location = get_transform_position(map_data["vehicle_data"]["hero_car"]["position"], translation_vector, rotation_matrix)
starting_position = carla.Transform(
    carla.Location(x=starting_location[0], y=starting_location[1], z=starting_location[2] + 0.5), 
    carla.Rotation(pitch=0, yaw=(map_data["vehicle_data"]["hero_car"]["heading"] + ROTATION_DEGREES) % 360, roll=0)
)

update_synchronous_mode(world, tm, True, model_data["camera"]["fps"])

# SPAWN HERO
vehicle_bp = blueprint_library.find(HERO_VEHICLE_TYPE)
vehicle = world.spawn_actor(vehicle_bp, starting_position)
vehicle.set_autopilot(False)

# SPAWN SENSORS
sensor_spawn_point = carla.Transform(carla.Location(x=model_data["camera"]["position"]["x"], y=model_data["camera"]["position"]["y"], z=model_data["camera"]["position"]["z"]), carla.Rotation(pitch=0.0))

seg_bp = get_sensor_blueprint(blueprint_library, 'sensor.camera.semantic_segmentation', model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"])
rgb_bp = get_sensor_blueprint(blueprint_library, 'sensor.camera.rgb', model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"])
depth_bp = get_sensor_blueprint(blueprint_library, 'sensor.camera.depth', model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"])
inst_bp = get_sensor_blueprint(blueprint_library, 'sensor.camera.instance_segmentation', model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"])
flow_bp = get_sensor_blueprint(blueprint_library, 'sensor.camera.optical_flow', model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"])

seg_sensor, seg_queue = setup_sensor(seg_bp, sensor_spawn_point, vehicle, world)
rgb_sensor, rgb_queue = setup_sensor(rgb_bp, sensor_spawn_point, vehicle, world)
depth_sensor, depth_queue = setup_sensor(depth_bp, sensor_spawn_point, vehicle, world)
inst_sensor, inst_queue = setup_sensor(inst_bp, sensor_spawn_point, vehicle, world)
flow_sensor, flow_queue = setup_sensor(flow_bp, sensor_spawn_point, vehicle, world)

world.tick()

# VEHICLE LIBRARY
vehicle_library = []
all_vehicles = blueprint_library.filter('vehicle.*')
forbidden_keywords = ["mustang", "police", "impala", "carlacola", "cybertruck", "t2", "sprinter", "firetruck", "ambulance", "bus", "truck", "van", "bingle", "microlino", "vespa", "yamaha", "kawasaki", "harley", "bh", "gazelle", "diamondback", "crossbike", "century", "omafiets", "low_rider", "ninja", "zx125", "yzf", "fuso", "rosa", "isetta"]
for bp in all_vehicles:
    if bp.has_attribute('number_of_wheels'):
        if int(bp.get_attribute('number_of_wheels')) != 4: continue
    if bp.has_attribute('generation'):
        if int(bp.get_attribute('generation')) != 1: continue
    if any(k in bp.id.lower() for k in forbidden_keywords): continue
    vehicle_library.append(bp)

spawn_positions = map_data["vehicle_data"]["spawn_positions"]
spectator = world.get_spectator()
pygame_screen, pygame_clock = setup_pygame(model_data["size"]["x"], model_data["size"]["y"], True)

output_dir = None
if not args.no_save:
    output_dir = args.output_dir if args.output_dir else "output_" + str(int(time.time()))
    output_dir = os.path.join(OUTPUT_FOLDER_NAME, output_dir)
    create_output_folders(output_dir)
    save_arguments(args, output_dir)
    flow_dir = os.path.join(output_dir, "flow_npy")
    instance_dir = os.path.join(output_dir, "instance")
    os.makedirs(flow_dir, exist_ok=True)
    os.makedirs(instance_dir, exist_ok=True)

# ==============================================================================
# ⚡ ROBUST MAPPING LOOP
# ==============================================================================
os.makedirs("mapping", exist_ok=True)
count = 0
detected_color_map = {} 
known_colors = set()

print("Initializing: Calibrating sensors and ignoring Hero Car...")
for _ in range(20):
    world.tick()
    pygame.event.pump() 
    flush_all_queues(seg_queue, rgb_queue, depth_queue, inst_queue, flow_queue)
    
    try:
        inst_data = inst_queue.get(timeout=1.0)
        initial_colors = get_unique_colors_from_sensor(inst_data)
        known_colors.update(initial_colors)
    except:
        pass

print(f"Calibration Complete. Ignored Colors: {len(known_colors)}")

for entry in spawn_positions:

    # ---------------------------------------------------------
    # LIMIT CHECK (Optional)
    # ---------------------------------------------------------
    if count >= 10:
        print("🛑 DEBUG LIMIT REACHED")
        break
    
    # --- MEMORY MANAGEMENT ---
    if count % 10 == 0:
        gc.collect() 
    
    current_actor = None
    actor_spawned = False
    
    # =========================================================
    # PHASE 1: SPAWN IN FRONT OF HERO
    # =========================================================
    spawn_retries = 0
    while not actor_spawned:
        pygame.event.pump() 
        
        spawned_actors = spawn_parked_cars_front_of_hero(
            world, vehicle_library, entry, 
            map_data["vehicle_data"]["hero_car"], 
            translation_vector, rotation_matrix
        )
        
        if spawned_actors:
            current_actor = spawned_actors[0]
            actor_spawned = True
        else:
            world.tick()
            flush_all_queues(seg_queue, rgb_queue, depth_queue, inst_queue, flow_queue)
            spawn_retries += 1
            if spawn_retries > 100:
                print(f"❌ Stuck on car {count}, skipping...")
                break 

    if not actor_spawned:
        count += 1
        continue

    # =========================================================
    # PHASE 2: MAP COLOR (Sensor Detection)
    # =========================================================
    detected_key = None
    attempts = 0
    max_attempts = 50 
    
    while attempts < max_attempts:
        world.tick()
        pygame.event.pump() 
        
        while not seg_queue.empty(): seg_queue.get()
        while not rgb_queue.empty(): rgb_queue.get()
        while not depth_queue.empty(): depth_queue.get()
        while not flow_queue.empty(): flow_queue.get()
        
        inst_data = inst_queue.get() 

        current_frame_colors = get_unique_colors_from_sensor(inst_data)
        new_colors = current_frame_colors - known_colors
        
        if new_colors:
            detected_key = list(new_colors)[0]
            detected_color_map[detected_key] = entry["color"]
            known_colors.add(detected_key)
            print(f"✅ Mapped Vehicle {count}: Color {detected_key}")
            
            # debug_img = process_instance_map_fixed(inst_data)
            # debug_img.save(os.path.join("mapping", f"debug_{count:03d}.png"))
            # del debug_img 
            break 
        
        del inst_data 
        attempts += 1
    
    if not detected_key:
        print(f"⚠️ Warning: TIMEOUT waiting for color for vehicle {count}")

    # =========================================================
    # PHASE 3: TELEPORT TO REAL POSITION
    # =========================================================
    start = get_transform_position(entry["start"], translation_vector, rotation_matrix)
    end = get_transform_position(entry["end"], translation_vector, rotation_matrix)
    base_z = 0.2
    print(base_z)
    heading = (entry["heading"] + ROTATION_DEGREES) % 360
    dx = end[0] - start[0]
    dy = end[1] - start[1]
        #length = math.sqrt(dx ** 2 + dy ** 2)
    length = 1
    ux, uy = dx / length, dy / length
    # Calculation of position
    x = start[0] 
    y = start[1]
    if len(start) > 2: 
        z = start[2]
    else: 
        print("0.5") 
        z=0.5
    print(x,y,z)

    veh_heading = heading
    if entry["mode"].strip() == "perpendicular" and random.random() < 0.5:
        veh_heading = (veh_heading + 180) % 360

    real_transform = carla.Transform(
        carla.Location(x=start[0], y=start[1], z=base_z+0.05), 
        carla.Rotation(yaw=veh_heading)
    )
    
    # 1. Teleport first
    
    try:
        blueprint = random.choice(vehicle_library)
        actor=world.spawn_actor(blueprint, real_transform)
        actor.destroy()
        current_actor.set_transform(real_transform)
        current_actor.set_simulate_physics(False)
        print(f"Spawned Car")

    except RuntimeError as e:
        current_actor.destroy()
        print(f"❌ Could not spawn carsas: {e}")
    # 2. Tick world to register new position internally (handles "rendering time")
    world.tick()
    flush_all_queues(seg_queue, rgb_queue, depth_queue, inst_queue, flow_queue)

    # =========================================================
    # PHASE 4: CHECK COLLISION & HANDLE
    # =========================================================
    


    count += 1

print("✅ All vehicles processed.")
print(f"Total Mapped: {len(detected_color_map)}")

# ==============================================================================
#  GRACEFUL HANDOFF
# ==============================================================================
file_path = "instance_map.txt"
json_ready_map = {str(k): v for k, v in detected_color_map.items()}

with open(file_path, 'w') as f:
    json.dump(json_ready_map, f, indent=4)

print(f"Saved successfully to {file_path}")

# --- STOP DATA STREAMS ---
if 'seg_sensor' in locals() and seg_sensor is not None: seg_sensor.stop()
if 'rgb_sensor' in locals() and rgb_sensor is not None: rgb_sensor.stop()
if 'depth_sensor' in locals() and depth_sensor is not None: depth_sensor.stop()
if 'inst_sensor' in locals() and inst_sensor is not None: inst_sensor.stop()
if 'flow_sensor' in locals() and flow_sensor is not None: flow_sensor.stop()

# --- CLEANUP PYTHON MEMORY ---
seg_queue = None
rgb_queue = None
depth_queue = None
inst_queue = None
flow_queue = None
gc.collect()

# --- FREEZE SIMULATION ---
print("-" * 30)
print("❄️  SIMULATION PAUSED (Synchronous Mode Active)")
print("🚗 Hero Vehicle & Parked Cars are frozen in position.")
print("-" * 30)
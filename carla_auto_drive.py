import carla
import time
import numpy as np
import pygame
import os
import json
import cv2 
from PIL import Image # <--- REQUIRED FOR COLOR MAPPING
from queue import Queue, Empty

# --- CONFIGURATION ---
CARLA_IP = '127.0.0.1'
CARLA_PORT = 2000
IM_WIDTH = 800
IM_HEIGHT = 600
OUTPUT_FOLDER = "dataset_output"

# Create directories
os.makedirs(os.path.join(OUTPUT_FOLDER, "rgb"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "semantic"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "instance"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "data"), exist_ok=True)

def remap_segmentation_colors(seg_image):
    arr = np.array(seg_image)
    # 0,0,0 → 128,64,128
    mask1 = np.all(arr == [0, 0, 0], axis=-1)
    arr[mask1] = [128, 64, 128]
    # 70,130,180 → 0,0,0
    mask2 = np.all(arr == [70, 130, 180], axis=-1)
    arr[mask2] = [0, 0, 0]
    return Image.fromarray(arr)
# ==============================================================================
#  COLOR MAPPING HELPER (EXACT COPY FROM SCRIPT A)
# ==============================================================================
def process_instance_map_fixed(inst_raw_data):
    """
    Parses CARLA Instance Bitmap.
    Red Channel = Category Tag.
    Green/Blue  = Instance ID.
    
    Refers to provided Table:
    14 = Car, 15 = Truck, 16 = Bus, 18 = Motorcycle
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
    
    # 5. Generate Colors (Hash logic from Script A)
    out_r = (g_id * 37 + b_id * 13) % 200 + 55
    out_g = (g_id * 17 + b_id * 43) % 200 + 55
    out_b = (g_id * 29 + b_id * 53) % 200 + 55
    
    # 6. Apply Colors ONLY to Vehicles
    output[vehicle_mask, 0] = out_r[vehicle_mask]
    output[vehicle_mask, 1] = out_g[vehicle_mask]
    output[vehicle_mask, 2] = out_b[vehicle_mask]
    
    return Image.fromarray(output)

# ==============================================================================
#  MAIN LOGIC
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

def save_data(frame_id, rgb_obj, sem_obj, inst_obj, transform):
    """
    Saves the data from the current tick to disk.
    Matches the format of Script A.
    """
    filename = f"{frame_id:06d}"
    
    # 1. Save RGB
    rgb_array = np.frombuffer(rgb_obj.raw_data, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (rgb_obj.height, rgb_obj.width, 4))
    rgb_array = rgb_array[:, :, :3] # Drop Alpha
    cv2.imwrite(f"{OUTPUT_FOLDER}/rgb/{filename}.png", rgb_array)

    # 2. Save Semantic Segmentation (MATCHING SCRIPT A FORMAT)
    # We convert to CityScapesPalette to get the colors (purple road, blue car, etc.)
    # 2. Save Semantic Segmentation
    # Convert to CityScapesPalette
    sem_obj.convert(carla.ColorConverter.CityScapesPalette)
    
    # Use the utility pipeline from Script A to remap colors exactly
    sem_pil = remap_segmentation_colors(carla_image_to_pil(sem_obj))
    
    # Save directly using the PIL image from the utility
    sem_pil.save(f"{OUTPUT_FOLDER}/semantic/{filename}.png")


    # 3. Save Instance Segmentation (MATCHING SCRIPT A LOGIC)
    # We use the fixed helper function to ensure colors match the mapping script
    inst_pil_image = process_instance_map_fixed(inst_obj)
    inst_pil_image.save(f"{OUTPUT_FOLDER}/instance/{filename}.png")

    # 4. Save Coordinates (Transform)
    data = {
        "frame": frame_id,
        "location": {
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z
        },
        "rotation": {
            "pitch": transform.rotation.pitch,
            "yaw": transform.rotation.yaw,
            "roll": transform.rotation.roll
        }
    }
    
    with open(f"{OUTPUT_FOLDER}/data/{filename}.json", 'w') as f:
        json.dump(data, f, indent=4)

def main():
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
    sensor_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15))

    # 1. RGB
    rgb_bp = bp_lib.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', str(IM_WIDTH))
    rgb_bp.set_attribute('image_size_y', str(IM_HEIGHT))
    rgb_bp.set_attribute('fov', '90')
    rgb_sensor = world.spawn_actor(rgb_bp, sensor_transform, attach_to=hero_vehicle)
    rgb_queue = Queue()
    rgb_sensor.listen(lambda data: sensor_callback(data, rgb_queue))

    # 2. Semantic Segmentation
    sem_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    sem_bp.set_attribute('image_size_x', str(IM_WIDTH))
    sem_bp.set_attribute('image_size_y', str(IM_HEIGHT))
    sem_bp.set_attribute('fov', '90')
    sem_sensor = world.spawn_actor(sem_bp, sensor_transform, attach_to=hero_vehicle)
    sem_queue = Queue()
    sem_sensor.listen(lambda data: sensor_callback(data, sem_queue))

    # 3. Instance Segmentation
    inst_bp = bp_lib.find('sensor.camera.instance_segmentation')
    inst_bp.set_attribute('image_size_x', str(IM_WIDTH))
    inst_bp.set_attribute('image_size_y', str(IM_HEIGHT))
    inst_bp.set_attribute('fov', '90')
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
            save_data(frame_count, rgb_data, sem_data, inst_data, current_transform)
            
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
        
        settings.synchronous_mode = False
        world.apply_settings(settings)
        pygame.quit()
        print(f"👋 Done. Saved {frame_count} frames.")

if __name__ == '__main__':
    main()
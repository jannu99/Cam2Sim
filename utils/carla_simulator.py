import random
import carla
import xml.etree.ElementTree as ET

import math
import numpy as np
from PIL import Image

import random
# from create_vehicle_data_from_centroids import odom_xy_to_latlon  # se è in un file diverso, dimmelo e te lo sistemo
from config import VERTEX_DISTANCE, MAX_ROAD_LENGTH, WALL_HEIGHT, EXTRA_WIDTH, ROTATION_DEGREES, CAR_SPACING, \
    FORWARDS_PARKING_PROBABILITY


def get_map_size(xodr_data):
    root = ET.fromstring(xodr_data)
    header = root.find("header")

    north = float(header.attrib.get("north", "0"))
    south = float(header.attrib.get("south", "0"))
    east = float(header.attrib.get("east", "0"))
    west = float(header.attrib.get("west", "0"))

    width = east - west
    height = north - south

    return width, height

def spawn_additional_vehicles(world, vehicle_blueprints, vehicle_amount):
    if vehicle_amount <= 0:
        return
    spawn_points = world.get_map().get_spawn_points()
    for i in range(0,vehicle_amount):
        world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

    for vehicle in world.get_actors().filter('*vehicle*'):
        vehicle.set_autopilot(True)

def update_synchronous_mode(world, tm, bool, fps = 1):
    settings = world.get_settings()
    if bool:
        settings.fixed_delta_seconds = 1 / fps
    settings.synchronous_mode = bool
    world.apply_settings(settings)

    tm.set_synchronous_mode(bool)

def remap_segmentation_colors(seg_image):
    arr = np.array(seg_image)
    # 0,0,0 → 128,64,128
    mask1 = np.all(arr == [0, 0, 0], axis=-1)
    arr[mask1] = [128, 64, 128]
    # 70,130,180 → 0,0,0
    mask2 = np.all(arr == [70, 130, 180], axis=-1)
    arr[mask2] = [0, 0, 0]
    return Image.fromarray(arr)

def carla_image_to_pil(image) -> Image.Image:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    rgb_array = array[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB
    return Image.fromarray(rgb_array)

def remove_sensor(sensor):
    if sensor is not None:
        sensor.stop()
        sensor.destroy()

def delete_all_vehicles(world):
    actors = world.get_actors().filter('vehicle.*')
    for actor in actors:
        actor.destroy()

def get_spectator_transform(world):
    spectator = world.get_spectator()
    transform = spectator.get_transform()
    return transform

def set_spectator_transform(world, transform):
    spectator = world.get_spectator()
    spectator.set_transform(transform)

def generate_world_map(client, xodr_data):
    return client.generate_opendrive_world(
        xodr_data,
        carla.OpendriveGenerationParameters(
            vertex_distance=VERTEX_DISTANCE,
            max_road_length=MAX_ROAD_LENGTH,
            wall_height=WALL_HEIGHT,
            additional_width=EXTRA_WIDTH,
            smooth_junctions=True,
            enable_mesh_visibility=True
        )
    )

def get_rotation_matrix(roation_degrees):
    theta = math.radians(roation_degrees)
    return np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])

def get_translation_vector(xodr_data, offset):
    """ Gibt den Translationsvektor für die Verschiebung in X, Y zurück. """
    size_x, size_y = get_map_size(xodr_data)

    center = np.array([offset["x"], offset["y"]])
    rotation_matrix = get_rotation_matrix(offset["heading"])
    rotated_center = rotation_matrix @ center

    translation = -rotated_center

    #print("ORIGINAL:", center)
    #print("ROTATED: ", rotated_center)
    #print("TRANSLATION VECTOR: ",translation)

    #return np.array(rotated_center[0], -rotated_center[1])
    #return np.array([292.5920715332031, -226.9042205810547])  # Verschiebung in X, Y (nach der Rotation)
    #return np.array([size_x / 2 + 5.42, size_y / -2 + 50.57])  # Verschiebung in X, Y (nach der Rotation)
    #return np.array([size_x / 2, size_y / -2])  # Verschiebung in X, Y (nach der Rotation)
    #print(np.array([size_x / 2 + translation[0], size_y / -2 + translation[1]]))
    return np.array([size_x / 2 + translation[0], size_y / -2 + translation[1]])  # Verschiebung in X, Y (nach der Rotation)

def get_transform_position(pos, translation_vector, rotation_matrix):
    """ wendet Rotation + Translation auf eine [x, y, z] Position an """
    xy = np.array(pos[:2])
    rotated = rotation_matrix.dot(xy)
    translated = rotated + translation_vector
    return [translated[0], translated[1], pos[2]]

def spawn_parked_cars(world, vehicle_library, spawn_positions, translation_vector, rotation_matrix):
    for entry in spawn_positions:
        car_id=entry["cluster_id"]
        start = get_transform_position(entry["start"], translation_vector, rotation_matrix)
        end = get_transform_position(entry["end"], translation_vector, rotation_matrix)
        heading = (entry["heading"] + ROTATION_DEGREES) % 360

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx ** 2 + dy ** 2)

        if length < 0.1:
            print("⚠️ Skipping very short segment.")
            continue

        ux, uy = dx / length, dy / length
        num_cars = int(length // CAR_SPACING)

        if num_cars == 0:
            print("⚠️ Too short for parked car.")
            continue

        for i in range(num_cars):
            blueprint = random.choice(vehicle_library)
            #blueprint = blueprint_library.find('vehicle.tesla.model3')

            x = start[0] + i * CAR_SPACING * ux
            y = start[1] + i * CAR_SPACING * uy
            z = start[2] if len(start) > 2 else 0.0

            veh_heading = heading
            print(entry["mode"] )
            if entry["mode"].strip() == "perpendicular" and random.random() < ( 50 / 100 ):
                veh_heading = (veh_heading + 180) % 360

            location = carla.Location(x=x, y=y, z=z)
            rotation = carla.Rotation(yaw=veh_heading)
            transform = carla.Transform(location, rotation)

            try:
                actor = world.spawn_actor(blueprint, transform)
                actor.set_simulate_physics(False)
                actor.set_target_velocity(carla.Vector3D(0, 0, 0))
                actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                #print(f"✅ Parked car at ({x:.1f}, {y:.1f}) heading {heading:.1f}°")
            except RuntimeError as e:
                print(f"❌ Could not spawn car: ({car_id:.1f}) at ({x:.1f}, {y:.1f}): {e}")

import carla
import random
import math

def spawn_parked_cars2(world, vehicle_library, spawn_positions, translation_vector, rotation_matrix):
    spawned_actors = [] 
    color_mapping = {} # Key: Actor ID (int), Value: Color [R, G, B]
    print("trzing for ")
    print(  len(spawn_positions))
    for entry in spawn_positions:
        print("1")
        # 1. Extract color from JSON
        target_color = None
        if "color" in entry and entry["color"]:
            try:
                target_color = [int(x) for x in entry["color"].split(',')]
            except:
                pass
        print("2")
        
        # Default to a standard grey if no color specified
        if target_color is None:
            target_color = [128, 128, 128]

        # Standard Transform math...
        start = get_transform_position(entry["start"], translation_vector, rotation_matrix)
        end = get_transform_position(entry["end"], translation_vector, rotation_matrix)
        heading = (entry["heading"] + ROTATION_DEGREES) % 360

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        #length = math.sqrt(dx ** 2 + dy ** 2)
        length = 1

        print("3")

        if length < 1: continue

        ux, uy = dx / length, dy / length
        num_cars = 1

        if num_cars == 0: continue

        print("4")

        for i in range(num_cars):
            blueprint = random.choice(vehicle_library)
            
            # --- CRITICAL STEP 1: Set Visual Color on the Car ---
            if blueprint.has_attribute('color'):
                # CARLA expects "255,0,0" string format
                color_str = f"{target_color[0]},{target_color[1]},{target_color[2]}"
                blueprint.set_attribute('color', color_str)

            # Calculation of position
            x = start[0] + i * CAR_SPACING * ux
            y = start[1] + i * CAR_SPACING * uy
            z = start[2] if len(start) > 2 else 0.0

            veh_heading = heading
            if entry["mode"].strip() == "perpendicular" and random.random() < 0.5:
                veh_heading = (veh_heading + 180) % 360

            transform = carla.Transform(
                carla.Location(x=x, y=y, z=z), 
                carla.Rotation(yaw=veh_heading)
            )

            try:
                actor = world.spawn_actor(blueprint, transform)
                actor.set_simulate_physics(False)
                print(f"Spawned Car")
                
                spawned_actors.append(actor)
                
                # --- CRITICAL STEP 2: Save the ID and Color for the Instance Map ---
                color_mapping[actor.id] = target_color
                
            except RuntimeError as e:
                print(f"❌ Could not spawn car: {e}")
    
    return spawned_actors, color_mapping

def spawn_parked_cars_and_map(world, vehicle_library, spawn_positions, translation_vector, rotation_matrix):
    """
    1. Spawns car at parking spot (checks collision).
    2. Teleports to Sky -> Scans ID -> Maps to JSON Color.
    3. Teleports back to parking spot.
    """
    spawned_actors = [] 
    color_mapping = {} # The Result: { InstanceID : [R, G, B] }
    CAR_SPACING = 5.0

    print(f"🔄 Processing {len(spawn_positions)} parking entries with Sky-Scanner...")

    # --- SETUP SCANNER (ISOLATION BOOTH) ---
    # We place a camera high up in the sky (Z=500) looking down
    bp_lib = world.get_blueprint_library()
    scanner_bp = bp_lib.find('sensor.camera.instance_segmentation')
    scanner_bp.set_attribute('image_size_x', '100')
    scanner_bp.set_attribute('image_size_y', '100')
    scanner_bp.set_attribute('fov', '90')
    scanner_bp.set_attribute('sensor_tick', '0.0')

    # Isolation coordinates
    ISO_X, ISO_Y, ISO_Z = 0, 0, 500
    
    # Spawn Camera 5m above the isolation point
    scanner_sensor = world.spawn_actor(scanner_bp, carla.Transform(
        carla.Location(x=ISO_X, y=ISO_Y, z=ISO_Z + 5.0),
        carla.Rotation(pitch=-90)
    ))
    
    scanner_queue = queue.Queue()
    scanner_sensor.listen(scanner_queue.put)

    try:
        for entry in spawn_positions:
            # 1. Parse JSON Color
            target_color = [128, 128, 128] # Default
            if "color" in entry and entry["color"]:
                try:
                    target_color = [int(x) for x in entry["color"].split(',')]
                except:
                    pass

            # 2. Calculate Parking Transform
            start = get_transform_position(entry["start"], translation_vector, rotation_matrix)
            end = get_transform_position(entry["end"], translation_vector, rotation_matrix)
            heading = (entry["heading"] + ROTATION_DEGREES) % 360

            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dist = math.sqrt(dx**2 + dy**2)
            length = dist if dist > 0 else 1 

            if length < 1: continue

            ux, uy = dx / length, dy / length
            
            # Logic for number of cars (simplified from your code)
            num_cars = 1 

            for i in range(num_cars):
                blueprint = random.choice(vehicle_library)
                
                # Set Visual Color (Visuals only)
                if blueprint.has_attribute('color'):
                    blueprint.set_attribute('color', f"{target_color[0]},{target_color[1]},{target_color[2]}")

                # Calc Position
                x = start[0] + i * CAR_SPACING * ux
                y = start[1] + i * CAR_SPACING * uy
                z = start[2] if len(start) > 2 else 0.2

                veh_heading = heading
                if "mode" in entry and entry["mode"].strip() == "perpendicular" and random.random() < 0.5:
                    veh_heading = (veh_heading + 180) % 360

                parking_transform = carla.Transform(
                    carla.Location(x=x, y=y, z=z), 
                    carla.Rotation(yaw=veh_heading)
                )

                # --- STEP A: TRY SPAWN AT PARKING SPOT ---
                # We do this first to check for collisions/validity
                actor = world.try_spawn_actor(blueprint, parking_transform)
                
                if actor is None:
                    # If we can't spawn here, just skip. No harm done.
                    continue
                
                actor.set_simulate_physics(False)
                spawned_actors.append(actor)

                # --- STEP B: TELEPORT TO SKY (ISOLATION) ---
                # Move car to x=0, y=0, z=500
                actor.set_transform(carla.Transform(carla.Location(x=ISO_X, y=ISO_Y, z=ISO_Z), carla.Rotation(yaw=0)))
                
                # --- STEP C: SNAPSHOT ---
                world.tick() # Render frame
                
                # Flush queue to get latest image
                image = None
                while not scanner_queue.empty():
                    image = scanner_queue.get()
                if image is None: image = scanner_queue.get(timeout=2.0)

                # --- STEP D: READ ID ---
                array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((100, 100, 4))
                center = array[45:55, 45:55] # Center crop
                
                b = center[:,:,0].astype(np.int32)
                g = center[:,:,1].astype(np.int32)
                ids = g + (b << 8)
                
                valid_ids = ids[ids > 0]
                
                if len(valid_ids) > 0:
                    detected_id = Counter(valid_ids.flatten()).most_common(1)[0][0]
                    # MAP IT: This ID = This JSON Color
                    color_mapping[detected_id] = target_color
                
                # --- STEP E: TELEPORT BACK HOME ---
                actor.set_transform(parking_transform)

    finally:
        if scanner_sensor: scanner_sensor.destroy()
        print(f"✅ Calibration Complete. Mapped {len(color_mapping)} vehicles.")
    
    return spawned_actors, color_mapping
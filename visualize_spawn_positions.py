import carla
import argparse
import json
import os
import time
import random
import numpy as np
from dotenv import load_dotenv

# Import your existing config/utils
from config import CARLA_IP, CARLA_PORT, ROTATION_DEGREES
from utils.save_data import get_map_data
# We need these specific functions to calculate the offset correctly
from utils.carla_simulator import (
    generate_world_map, 
    get_rotation_matrix, 
    get_translation_vector, 
    get_transform_position
)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Spawn Positions from JSON")
    parser.add_argument("--map", type=str, required=True, help="Name of the map folder (inside maps/)")
    parser.add_argument("--json", type=str, required=True, help="Path to the generated vehicle_data.json")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Connect to CARLA
    print(f"🔌 Connecting to CARLA at {CARLA_IP}:{CARLA_PORT}...")
    client = carla.Client(CARLA_IP, CARLA_PORT)
    client.set_timeout(20.0)

    # 2. Load Map Data & Offset
    print(f"🗺️  Loading Map: {args.map}...")
    # This loads the map's original vehicle_data.json which contains the OFFSET
    map_data = get_map_data(args.map, (100, 100)) 
    
    if map_data is None:
        print(f"❌ CRITICAL ERROR: Could not load map '{args.map}'. Check spelling.")
        return

    # EXTRACT OFFSET DATA
    # This is the "fixed information" you mentioned
    offset_data = map_data["vehicle_data"]["offset"]
    print(f"📐 Found Map Offset: x={offset_data['x']}, y={offset_data['y']}")

    # PREPARE TRANSFORMATION MATRICES
    # This logic matches your pipeline script exactly
    rotation_matrix = get_rotation_matrix(ROTATION_DEGREES)
    translation_vector = get_translation_vector(map_data["xodr_data"], offset_data)

    # Load the OpenDRIVE map
    world = generate_world_map(client, map_data["xodr_data"])
    blueprint_library = world.get_blueprint_library()

    # 3. Load Generated Spawn Positions
    if not os.path.exists(args.json):
        raise FileNotFoundError(f"JSON file not found: {args.json}")

    print(f"📂 Loading Spawn Data from: {args.json}")
    with open(args.json, 'r') as f:
        data = json.load(f)
    
    spawn_list = data.get("spawn_positions", [])
    print(f"found {len(spawn_list)} positions to spawn.")

    # 4. Filter Vehicles
    vehicle_bp_list = []
    all_vehicles = blueprint_library.filter('vehicle.*')
    forbidden = ["truck", "van", "bus", "police", "fire", "ambulance", "carlacola", "cybertruck", "sprinter", "microlino", "vespa", "yamaha", "kawasaki", "harley"]
    
    for bp in all_vehicles:
        if bp.has_attribute('number_of_wheels'):
            if int(bp.get_attribute('number_of_wheels')) != 4: continue
        if any(k in bp.id.lower() for k in forbidden): continue
        vehicle_bp_list.append(bp)

    spawned_actors = []

    try:
        print("🚗 Spawning vehicles...")
        
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        for item in spawn_list:
            # Get the Raw Point
            raw_loc = item.get("start") 
            heading = item.get("heading", 0.0)

            if raw_loc is None: continue

            # --- APPLY OFFSET TRANSFORMATION ---
            # This moves the point from "Raw Map Coordinates" to "CARLA World Coordinates"
            # using the offset we loaded from the map data.
            final_pos = get_transform_position(raw_loc, translation_vector, rotation_matrix)

            # CARLA Transform (x, y, z)
            # z + 0.5 to prevent clipping
            carla_loc = carla.Location(x=final_pos[0], y=final_pos[1], z=final_pos[2] + 0.5)
            
            # Apply Heading Rotation
            # We add ROTATION_DEGREES because the whole map might be rotated
            final_yaw = (heading + ROTATION_DEGREES) % 360
            carla_rot = carla.Rotation(pitch=0.0, yaw=final_yaw, roll=0.0)
            
            transform = carla.Transform(carla_loc, carla_rot)

            bp = random.choice(vehicle_bp_list)
            vehicle = world.try_spawn_actor(bp, transform)
            
            if vehicle:
                vehicle.set_simulate_physics(False) 
                spawned_actors.append(vehicle)

        print(f"✅ Successfully spawned {len(spawned_actors)} vehicles.")

        if spawned_actors:
            spectator = world.get_spectator()
            first_car_trans = spawned_actors[0].get_transform()
            spectator.set_transform(carla.Transform(
                first_car_trans.location + carla.Location(z=50),
                carla.Rotation(pitch=-90)
            ))

        print("Press Ctrl+C to exit and destroy actors.")
        
        while True:
            world.wait_for_tick()

    except KeyboardInterrupt:
        print("\nCleaning up...")
    finally:
        print(f"🗑️ Destroying {len(spawned_actors)} actors...")
        client.apply_batch([carla.command.DestroyActor(x) for x in spawned_actors])
        print("Done.")

if __name__ == "__main__":
    main()
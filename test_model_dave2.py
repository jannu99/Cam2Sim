import carla
import time
import os
import numpy as np
import cv2
from collections import Counter
from PIL import Image, ImageOps 
import torchvision.transforms as transforms
import hashlib # Necessario per la funzione hash, anche se usiamo una formula semplice

from dotenv import load_dotenv
from huggingface_hub import login
from config import HERO_VEHICLE_TYPE, CARLA_IP, CARLA_PORT, ROTATION_DEGREES, OUTPUT_FOLDER_NAME
from utils.argparser import parse_testing_args
from utils.carla_simulator import update_synchronous_mode, remap_segmentation_colors, \
    carla_image_to_pil, remove_sensor, generate_world_map, get_rotation_matrix, \
    get_translation_vector, spawn_parked_cars2, get_transform_position
from utils.pygame_helper import combine_images, setup_pygame, setup_sensor, get_sensor_blueprint, \
    show_image
from utils.save_data import create_dotenv, get_map_data, create_output_folders, get_model_data, save_arguments
from utils.dave2_connection import connect_to_dave2_server, send_image_over_connection
from utils.stable_diffusion import load_stable_diffusion_pipeline, generate_image
from utils.distortion import compute_intrinsic_matrix, simulate_distortion_from_pinhole
from utils.yolo import calculate_yolo_image, load_yolo_model

# === NUOVA FUNZIONE DI UTILITÀ (Genera la Mappa Hash/ID) ===
def generate_instance_hash_image(inst_array_3ch):
    """
    Decodifica i bit dell'ID e genera un'immagine colorata con un colore casuale
    determinato dall'ID (utile per isolare le maschere).
    """
    # Usiamo la formula G + B*256, che produce l'ID (se non l'offset).
    b, g, r = inst_array_3ch[:,:,0].astype(np.int32), inst_array_3ch[:,:,1].astype(np.int32), inst_array_3ch[:,:,2].astype(np.int32)
    pixel_ids_map = g + (b << 8)
    
    # 1. Base Layer: Genera colori hash consistenti per ogni ID
    r_hash = (pixel_ids_map * 13) % 255
    g_hash = (pixel_ids_map * 37) % 255
    b_hash = (pixel_ids_map * 61) % 255
    hashed_inst = np.dstack((r_hash, g_hash, b_hash)).astype(np.uint8)

    return hashed_inst, pixel_ids_map

# === FUNZIONE CALCOLO MODA (COLORE PIÙ FREQUENTE) ===
def get_most_frequent_color(pixels):
    """Calcola la moda (colore più frequente) in un array di pixel RGB."""
    if len(pixels) == 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    
    # Converte i pixel RGB in un singolo int per contare i valori unici
    packed = pixels[:, 0] + (pixels[:, 1] << 8) + (pixels[:, 2] << 16)
    
    # Contatore standard per trovare la moda
    counts = Counter(packed)
    
    # Il valore più comune (la moda)
    most_common_packed = counts.most_common(1)[0][0]
    
    # Riconverte la moda in RGB (B, G, R)
    r = (most_common_packed) & 0xFF
    g = (most_common_packed >> 8) & 0xFF
    b = (most_common_packed >> 16) & 0xFF
    
    # Numpy array format
    return np.array([r, g, b], dtype=np.uint8) # Restituisce [R, G, B]


# --- ARGS & SETUP ---
args = parse_testing_args()

if not args.only_carla:
    create_dotenv()
    load_dotenv()
    huggingface_token = os.getenv("HF_TOKEN")
    if huggingthingface_token: login(token=huggingface_token)

client = carla.Client(args.carla_ip, args.carla_port)
client.set_timeout(40.0) 
tm = client.get_trafficmanager(8000)
world = client.get_world()

model_data = get_model_data(args.model)
map_data = get_map_data(args.map, (model_data["size"]["x"], model_data["size"]["y"]))
offset_data = map_data["vehicle_data"]["offset"]

# --- GENERATE MAP ---
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

# Sync Mode
update_synchronous_mode(world, tm, True, model_data["camera"]["fps"])

# 1. SPAWN HERO
vehicle_bp = blueprint_library.find(HERO_VEHICLE_TYPE)
vehicle = world.spawn_actor(vehicle_bp, starting_position)
#vehicle.set_autopilot(True)

# 2. SPAWN SENSORS
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

# 3. SPAWN PARKED CARS
vehicle_library = []
all_vehicles = blueprint_library.filter('vehicle.*')
forbidden_keywords = ["mustang", "police", "impala", "carlacola", "cybertruck", "t2", "sprinter", "firetruck", "ambulance", "bus", "truck", "van", "bingle", "microlino", "vespa", "yamaha", "kawasaki", "harley", "bh", "gazelle", "diamondback", "crossbike", "century", "omafiets", "low_rider", "ninja", "zx125", "yzf", "fuso", "rosa", "isetta"]
for bp in all_vehicles:
    if bp.has_attribute('number_of_wheels'):
        if int(bp.get_attribute('number_of_wheels')) != 4: continue
    # 3. NEW CHECK: Filter for Generation 1 only
    if bp.has_attribute('generation'):
        if int(bp.get_attribute('generation')) != 1:
            continue  # Skip if the generation is not 1
    if any(k in bp.id.lower() for k in forbidden_keywords): continue
    vehicle_library.append(bp)

initial_mapping, color_mapping = spawn_parked_cars2(world, vehicle_library, map_data["vehicle_data"]["spawn_positions"], translation_vector, rotation_matrix)
color_mapping[vehicle.id] = [255, 255, 255] # Hero car white

print(f"✅ Loaded Color Mapping for {len(color_mapping)} cars.")

spectator = world.get_spectator()
pygame_screen, pygame_clock = setup_pygame(model_data["size"]["x"], model_data["size"]["y"], args.only_carla)

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
dave2_conn = None
if not args.only_carla:
    pipe = load_stable_diffusion_pipeline(args.model, model_data)
    prev_image = map_data["starting_image"]
    yolo_model = load_yolo_model()

dave2_conn = connect_to_dave2_server()

image_transforms = transforms.Compose([transforms.Resize(model_data["size"]["x"], interpolation=transforms.InterpolationMode.BILINEAR), transforms.CenterCrop(model_data["size"]["x"])])
instance_transforms = transforms.Compose([transforms.Resize(model_data["size"]["x"], interpolation=transforms.InterpolationMode.NEAREST), transforms.CenterCrop(model_data["size"]["x"])])

frame = world.tick()
kill_frame = frame + (args.seconds * model_data["camera"]["fps"])

try:
    while True:
        frame = world.tick()
        spectator.set_transform(vehicle.get_transform()) 

        # GET DATA
        seg_data = seg_queue.get()
        rgb_data = rgb_queue.get()
        depth_data = depth_queue.get()
        inst_data = inst_queue.get()
        flow_data = flow_queue.get()

        flow_array = np.frombuffer(flow_data.raw_data, dtype=np.float32).reshape((flow_data.height, flow_data.width, 2))
        
        # RGB FRAME (Ground Truth for Color)
        rgb_image_np = np.frombuffer(rgb_data.raw_data, dtype=np.uint8).reshape((rgb_data.height, rgb_data.width, 4))[:,:,:3][:,:,::-1] # RGB Array (H, W, 3)
        rgb_image_pil = Image.fromarray(rgb_image_np)

        # INSTANCE FRAME (IDs to be masked)
        inst_array = np.frombuffer(inst_data.raw_data, dtype=np.uint8).reshape((inst_data.height, inst_data.width, 4))[:,:,:3] # Only RGB channels

        # 1. CREAZIONE MAPPA HASH CASUALE (Usata solo per creare le maschere/ID unici)
        hashed_inst_np, pixel_ids_map = generate_instance_hash_image(inst_array)
        
        # 2. ANALISI DEI COLORI PREDOMINANTI
        # Identifica tutti i colori/ID unici nella scena
        unique_colors, counts = np.unique(hashed_inst_np.reshape(-1, 3), axis=0, return_counts=True)
        
        # Ordina per frequenza (il primo è lo sfondo)
        sorted_indices = np.argsort(-counts)
        predominant_color = unique_colors[sorted_indices[0]] 
        
        # Inizializza l'output finale
        final_colored_inst = np.zeros_like(hashed_inst_np)
        
        # 3. MASKING E COLORAZIONE (Moda)
        for i in sorted_indices:
            color_hash = unique_colors[i]
            count = counts[i]
            
            # Crea una maschera per ISOLARE tutti i pixel di questo colore/ID
            is_target_mask = np.all(hashed_inst_np == color_hash, axis=-1)
            
            # Logica: 
            # 1. Se è il colore predominante (strada, cielo) o rumore (< 100 pixel)
            if np.array_equal(color_hash, predominant_color) or count < 100: 
                 # Colore predominante/Sfondo deve essere NERO (come richiesto)
                 final_colored_inst[is_target_mask] = [0, 0, 0] 
            else:
                # 2. Questo è un oggetto (un'auto, un palazzo, o un albero)
                
                # A. Trova i pixel RGB del frame originale sotto questa maschera
                original_pixels = rgb_image_np[is_target_mask]
                
                # B. Calcola il colore più frequente (MODA)
                if len(original_pixels) > 0:
                    true_color_gt = get_most_frequent_color(original_pixels)
                    
                    # C. Applica il Colore Più Frequente sulla Maschera
                    final_colored_inst[is_target_mask] = true_color_gt
                else:
                    final_colored_inst[is_target_mask] = [0, 0, 0] # Fallback nero
                 
        instance_pil = Image.fromarray(final_colored_inst)

        # ======================================================================
        #  REST OF PIPELINE
        # ======================================================================
        seg_data.convert(carla.ColorConverter.CityScapesPalette) 
        seg_image = remap_segmentation_colors(carla_image_to_pil(seg_data))

        if "calibration" in model_data["camera"].keys() :
            K_pinhole = compute_intrinsic_matrix(model_data["camera"]["original_size"]["x"], model_data["camera"]["original_size"]["y"], model_data["camera"]["fov"]) 
            K_real = np.array(model_data["camera"]["calibration"]["K"], dtype=np.float32) 
            dist_real = np.array(model_data["camera"]["calibration"]["distortion"], dtype=np.float32)
            distorted_seg = simulate_distortion_from_pinhole(seg_image, K_pinhole, K_real, dist_real)
            distorted_rgb = simulate_distortion_from_pinhole(rgb_image_pil, K_pinhole, K_real, dist_real)
            distorted_inst = simulate_distortion_from_pinhole(instance_pil, K_pinhole, K_real, dist_real, interpolation=cv2.INTER_NEAREST) 
            final_seg = image_transforms(distorted_seg) 
            final_rgb = image_transforms(distorted_rgb)
            final_inst = instance_transforms(distorted_inst)
        else:
            final_seg = image_transforms(seg_image)
            final_rgb = image_transforms(rgb_image_pil)
            final_inst = instance_transforms(instance_pil) 

        depth_data.convert(carla.ColorConverter.LogarithmicDepth) 
        depth_image_np = cv2.filter2D(np.array(carla_image_to_pil(depth_data)),-1,np.ones((3,3),np.float32)/10)
        final_depth = image_transforms(ImageOps.invert(Image.fromarray(depth_image_np)))

        combined = final_rgb
        steering_image = final_rgb
        if not args.only_carla:
            generated_image = generate_image(pipe, final_seg, model_data, prev_image, split = 30, guidance = 3.5, set_seed = True, rotate = False)
            prev_image = generated_image
            _, yolo_image = calculate_yolo_image(yolo_model, generated_image)
            combined = combine_images(final_rgb, yolo_image)
            steering_image = generated_image

        steering, throttle = send_image_over_connection(dave2_conn, steering_image)
        # prev_steer = steering

        # steering = prev_steer * 0.8 + steering * (1.0 - 0.8)
            #steering, throttle = calculate_dave2_image(dave2_model, generated_image)
        print(f"Steering: {steering}, SteeringRad: {np.radians(steering)}, Throttle: {throttle}")
        steering_rad = np.radians(steering)
        ackermann_control = carla.VehicleAckermannControl()
        ackermann_control.speed = float(12 / 3.6)   # Zielgeschwindigkeit in m/s
        ackermann_control.steer = float(steering * -1.0)
        ackermann_control.steer_speed = 0.0
        ackermann_control.acceleration = 0.0
        ackermann_control.jerk = 0.0
            
            #print(ackermann_control)
        vehicle.apply_ackermann_control(ackermann_control)



        if not args.no_save:
            final_seg.save(os.path.join(output_dir, "seg", '%06d.png' % frame))
            if not args.only_carla: generated_image.save(os.path.join(output_dir, "output", '%06d.png' % frame))
            final_rgb.save(os.path.join(output_dir, "carla", '%06d.png' % frame))
            final_depth.save(os.path.join(output_dir, "depth", '%06d.png' % frame))
            final_inst.save(os.path.join(instance_dir, '%06d.png' % frame))
            
            # np.save(os.path.join(flow_dir, '%06d.npy' % frame), flow_array)

        show_image(pygame_screen, final_rgb)
        pygame_clock.tick(30)
        if frame >= kill_frame: break
            
except KeyboardInterrupt:
    print("User interrupted.")
finally:
    if 'seg_sensor' in locals(): remove_sensor(seg_sensor)
    if 'rgb_sensor' in locals(): remove_sensor(rgb_sensor)
    update_synchronous_mode(world, tm, False)
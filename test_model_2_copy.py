import carla
import time
import os
import numpy as np
import cv2
from PIL import Image, ImageOps 
import torchvision.transforms as transforms

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
from utils.stable_diffusion import load_stable_diffusion_pipeline, generate_image
from utils.distortion import compute_intrinsic_matrix, simulate_distortion_from_pinhole
from utils.yolo import calculate_yolo_image, load_yolo_model

# ==============================================================================
#  ⚡ FIXED: MAPPING BASED ON YOUR TABLE (Tags 14, 15, 16, 18)
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
    # CARLA images are usually BGRA in memory
    b_id = bgra[:, :, 0].astype(np.int32) 
    g_id = bgra[:, :, 1].astype(np.int32)
    tag  = bgra[:, :, 2] # Red Channel is the Semantic Tag
    
    # 3. Create Mask for VEHICLES based on YOUR TABLE
    # 14: Car, 15: Truck, 16: Bus, 18: Motorcycle
    # We use np.isin to check against multiple values efficiently
    vehicle_mask = np.isin(tag, [14, 15, 16, 18])
    
    # 4. Initialize Output (Black Background)
    output = np.zeros((inst_raw_data.height, inst_raw_data.width, 3), dtype=np.uint8)
    
    # 5. Generate Colors
    # We scramble G and B to make IDs look distinct
    out_r = (g_id * 37 + b_id * 13) % 200 + 55
    out_g = (g_id * 17 + b_id * 43) % 200 + 55
    out_b = (g_id * 29 + b_id * 53) % 200 + 55
    
    # 6. Apply Colors ONLY to Vehicles
    output[vehicle_mask, 0] = out_r[vehicle_mask]
    output[vehicle_mask, 1] = out_g[vehicle_mask]
    output[vehicle_mask, 2] = out_b[vehicle_mask]
    
    return Image.fromarray(output)


# --- ARGS & SETUP ---
args = parse_testing_args()

if not args.only_carla:
    create_dotenv()
    load_dotenv()
    huggingface_token = os.getenv("HF_TOKEN")
    if huggingface_token: login(token=huggingface_token)

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
vehicle.set_autopilot(True)

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
    if bp.has_attribute('generation'):
        if int(bp.get_attribute('generation')) != 1: continue
    if any(k in bp.id.lower() for k in forbidden_keywords): continue
    vehicle_library.append(bp)

initial_mapping, color_mapping = spawn_parked_cars2(world, vehicle_library, map_data["vehicle_data"]["spawn_positions"], translation_vector, rotation_matrix)
color_mapping[vehicle.id] = [255, 255, 255] # Hero car white

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

if not args.only_carla:
    pipe = load_stable_diffusion_pipeline(args.model, model_data)
    prev_image = map_data["starting_image"]
    yolo_model = load_yolo_model()

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
        
        # RGB FRAME
        rgb_image_np = np.frombuffer(rgb_data.raw_data, dtype=np.uint8).reshape((rgb_data.height, rgb_data.width, 4))[:,:,:3][:,:,::-1]
        rgb_image_pil = Image.fromarray(rgb_image_np)

        # === ⚡ CALLING THE FIXED FUNCTION (TAGS 14-18) ===
        instance_pil = process_instance_map_fixed(inst_data)
        # =================================================

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
        if not args.only_carla:
            generated_image = generate_image(pipe, final_seg, model_data, prev_image, split = 30, guidance = 3.5, set_seed = True, rotate = False)
            prev_image = generated_image
            _, yolo_image = calculate_yolo_image(yolo_model, generated_image)
            combined = combine_images(final_rgb, yolo_image)

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
    if 'inst_sensor' in locals(): remove_sensor(inst_sensor)
    update_synchronous_mode(world, tm, False)
import carla
import numpy as np
from config import HERO_VEHICLE_TYPE, ROTATION_DEGREES, OUTPUT_FOLDER_NAME

from utils.carla_simulator import update_synchronous_mode, \
    generate_world_map, get_rotation_matrix, \
    get_translation_vector, get_transform_position
from utils.save_data import create_dotenv, get_map_data, create_output_folders, get_model_data, save_arguments



def get_inverse_transform(carla_transform, model, map_name):
    # ... [Imports and Data fetching remain the same] ...
    rotation_matrix = get_rotation_matrix(ROTATION_DEGREES)    
    model_data = get_model_data(model)
    map_data = get_map_data(map_name, (model_data["size"]["x"], model_data["size"]["y"]))
    offset_data = map_data["vehicle_data"]["offset"]
    translation_vector = get_translation_vector(map_data["xodr_data"], offset_data)

    # 1. Extract CARLA data
    c_loc = carla_transform.location
    c_rot = carla_transform.rotation

    # 2. Revert the manual Z offset
    adjusted_z = c_loc.z - 0.5
    
    # 3. Prepare vectors for Matrix Math
    point_world = np.array([c_loc.x, c_loc.y, adjusted_z]) 
    
    # --- MATRIX PREP ---
    t_vec = np.array(translation_vector)
    r_mat = np.array(rotation_matrix)

    # Safety Check: 2D -> 3D
    if t_vec.shape[0] == 2:
        t_vec = np.append(t_vec, 0) 

    if r_mat.shape == (2, 2):
        new_mat = np.eye(3)
        new_mat[:2, :2] = r_mat
        r_mat = new_mat
    # -------------------
    
    # 4. Perform Inverse Affine Transformation (Location)
    point_translated = point_world - t_vec
    point_original = np.dot(r_mat.T, point_translated)
    
    # 5. Revert Rotation (Yaw, Pitch, Roll)
    # Yaw: Remove the map rotation offset
    original_yaw = (c_rot.yaw - ROTATION_DEGREES) % 360
    
    # Pitch & Roll: These are local to the vehicle. 
    # Since we only rotated the map on the Z-axis (Yaw), Pitch and Roll are preserved.
    original_pitch = c_rot.pitch
    original_roll = c_rot.roll

    return {
        "location": {
            "x": point_original[0],
            "y": point_original[1],
            "z": point_original[2]
        },
        "rotation": {
            "yaw": original_yaw,
            "pitch": original_pitch,
            "roll": original_roll
        }
    }
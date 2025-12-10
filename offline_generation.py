import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from huggingface_hub import login, snapshot_download
from datasets import load_dataset
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from safetensors import safe_open
import random
import math
from typing import List, Tuple, Dict, Any
from itertools import product
from itertools import permutations, product
# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

HF_TOKEN = ""
MODEL_REPO = "jannu99/gurickestraemp4_25_12_02_model"
TRAINING_DATASET_NAME = "jannu99/gurickestraemp4_25_12_02"

# Paths
LOCAL_MODEL_DIR = "./gurickestraemp4_25_12_02_local"


# 🌟 MODIFICATO: Limita l'elaborazione ai primi 1000 frame
FRAMES_TO_PROCESS_N = 100 

# Pipeline Constants
STABLE_DIFF_STEPS = 50


# --- 2. DOWNLOAD MODELLI BASE & CONFIG ---
from typing import List, Dict, Union, Any, Tuple
from itertools import product

# Definisci il tipo per chiarezza
ControlDict = Dict[str, List[float]]




def generate_control_dicts() -> List[ControlDict]:
    """
    Crea una lista di 50 dizionari di controllo. Inizializza con i 9 schemi specificati
    dall'utente (hardcoded) e completa il resto fino a 50 con schemi casuali validi.
    """
    
    # --- 1. SET INIZIALE FISSO (9 SCHEMI FORNITI DALL'UTENTE) ---
    control_dicts_old: List[ControlDict] = [
        # Schemi 0.0 -> 1.0 forti (3)
        {
            'start': [0.0, 0.0, 0.0],
            'end': [1.0, 1.0, 1.0]
        },
        # Schemi Permutazioni 0.33 (6)
        {
            'start': [0.00, 0.34, 0.68], 'end': [0.33, 0.67, 1.00]
        },
        {
            'start': [0.00, 0.68, 0.34], 'end': [0.33, 1.00, 0.67]
        },
        {
            'start': [0.34, 0.00, 0.68], 'end': [0.67, 0.33, 1.00]
        },
        {
            'start': [0.34, 0.68, 0.00], 'end': [0.67, 1.00, 0.33]
        },
        {
            'start': [0.68, 0.00, 0.34], 'end': [1.00, 0.33, 0.67]
        },
        {
            'start': [0.68, 0.34, 0.00], 'end': [1.00, 0.67, 0.33]
        }
    ]

    control_dicts: List[ControlDict] = [
        # Schemi 0.0 -> 1.0 forti (3)
        {
            'start': [0.00, 0.00, 0.00],
            'end': [0.33, 0.33, 0.33]
        },
        {
            'start': [0.00, 0.00, 0.34],
            'end': [0.33, 0.33, 1.00]
        },
        {
            'start': [0.00, 0.34, 0.00],
            'end': [0.33, 1.00, 0.33]
        },
        {
            'start': [0.00, 0.34, 0.34],
            'end': [0.33, 1.00, 1.00]
        },
        {
            'start': [0.34, 0.00, 0.00],
            'end': [1.00, 0.33, 0.33]
        },
        {
            'start': [0.34, 0.00, 0.34],
            'end': [1.00, 0.33, 1.00]
        },
        {
            'start': [0.34, 0.34, 0.00],
            'end': [1.00, 1.00, 0.33]
        },
        {
            'start': [0.34, 0.34, 0.34],
            'end': [1.00, 1.00, 1.00]
        },
        {
            'start': [0.00, 0.00, 0.00],
            'end': [0.66, 0.66, 0.66]
        },
        {
            'start': [0.00, 0.00, 0.67],
            'end': [0.66, 0.66, 1.00]
        },
        {
            'start': [0.00, 0.67, 0.00],
            'end': [0.66, 1.00, 0.66]
        },
        {
            'start': [0.00, 0.67, 0.67],
            'end': [0.66, 1.00, 1.00]
        },
        {
            'start': [0.67, 0.00, 0.00],
            'end': [1.00, 0.66, 0.66]
        },
        {
            'start': [0.67, 0.00, 0.67],
            'end': [1.00, 0.66, 1.00]
        },
        {
            'start': [0.67, 0.67, 0.00],
            'end': [1.00, 1.00, 0.66]
        },
        {
            'start': [0.67, 0.67, 0.67],
            'end': [1.00, 1.00, 1.00]
        }
    ]


    return control_dicts


def check_essential_files(model_dir: str) -> bool:
    if not os.path.exists(model_dir):
        return False

    required_files = [
        "config.json",
        "stable_diffusion/pytorch_lora_weights.safetensors", 
        "controlnet_instance/diffusion_pytorch_model.safetensors",
    ]
    
    for rel_path in required_files:
        if not os.path.exists(os.path.join(model_dir, rel_path)):
            print(f"File mancante: {os.path.join(model_dir, rel_path)}")
            return False
            
    return True

def download_models_and_config() -> str:
    """Login and download only the essential files, or use cached version."""
    
    if check_essential_files(LOCAL_MODEL_DIR):
        print(f"✅ Modelli essenziali già trovati in {LOCAL_MODEL_DIR}. Saltando il download.")
        return LOCAL_MODEL_DIR
    
    print("⏳ Logging into Hugging Face...")
    login(token=HF_TOKEN, add_to_git_credential=False)

    allow_patterns = [
        "config.json",
        "lora_weights/*", 
        "stable_diffusion/*", 
        "controlnet_segmentation/diffusion_pytorch_model.safetensors",
        "controlnet_segmentation/config.json",
        "controlnet_instance/diffusion_pytorch_model.safetensors",
        "controlnet_instance/config.json",
        "controlnet_tempconsistency/diffusion_pytorch_model.safetensors",
        "controlnet_tempconsistency/config.json",
    ]

    print(f"⏳ Downloading essential files from model repository: {MODEL_REPO}")
    model_root = snapshot_download(
        repo_id=MODEL_REPO,
        repo_type="model",
        local_dir=LOCAL_MODEL_DIR,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns
    )
    print(f"✅ Essential files downloaded to: {model_root}")
    return model_root

# --- 3. LOAD PIPELINE MODELS (ControlNet & LoRA) ---

def load_pipeline_models(model_root: str) -> Tuple[StableDiffusionControlNetPipeline, Dict[str, Any], Tuple[int, int]]:
    """Load ControlNet models, LoRA weights, and create the pipeline."""
    # Carica Config
    config_path = os.path.join(model_root, "config.json")
    with open(config_path, "r") as f:
        model_data = json.load(f)

    IMG_W = model_data["size"]["x"]
    IMG_H = model_data["size"]["y"]

    seg_dir_name  = model_data["controlnet_segmentation"]
    temp_dir_name = model_data["controlnet_tempconsistency"]
    instance_dir_name = model_data["controlnet_instance"]

    controlnet_seg_path  = os.path.join(model_root, seg_dir_name)
    controlnet_temp_path = os.path.join(model_root, temp_dir_name)
    controlnet_instance_path = os.path.join(model_root, instance_dir_name)

    print("\n⏳ Loading ControlNet Models...")
    controlnet_seg = ControlNetModel.from_pretrained(controlnet_seg_path, torch_dtype=torch.float16)
    controlnet_temp = ControlNetModel.from_pretrained(controlnet_temp_path, torch_dtype=torch.float16)
    controlnet_instance = ControlNetModel.from_pretrained(controlnet_instance_path, torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_data["stable_diffusion_model"],
        controlnet=[controlnet_seg, controlnet_instance, controlnet_temp],
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)

    # Carica il LoRA come state_dict
    lora_path = os.path.join(model_root, model_data["lora_weights"])
    print(f"⏳ Loading LoRA state dict from: {lora_path}")
    
    if lora_path.endswith(".safetensors"):
        lora_state_dict = {}
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                lora_state_dict[key] = f.get_tensor(key)
    else:
        lora_state_dict = torch.load(lora_path, map_location="cpu")

    pipe.load_lora_weights(lora_state_dict) 

    print("✅ Pipeline Loaded successfully with 3 ControlNets and LoRA.")
    return pipe, model_data, (IMG_W, IMG_H)

# --- 4. LOAD TRAINING DATASET & PRE-PROCESS ---

def load_and_preprocess_dataset(
    target_resolution: Tuple[int, int], 
    indices: List[int]
) -> Tuple[List[Image.Image], List[Image.Image], List[str], List[int]]:
    """Load and preprocess only the specified frames from the dataset."""
    print(f"\n⏳ Loading and preprocessing dataset: {TRAINING_DATASET_NAME}")

    ds = load_dataset(TRAINING_DATASET_NAME, split="train")
    
    ds_selected = ds.select(indices)

    test_segmentations: List[Image.Image] = []
    test_instances: List[Image.Image] = []
    test_captions: List[str] = []
    
    original_indices: List[int] = indices 

    print(f"⏳ Preprocessing {len(ds_selected)} selected examples, resizing to {target_resolution}...")

    for item in tqdm(ds_selected):
        seg_img = item['segmentation'].convert("RGB").resize(target_resolution)
        inst_img = item['instance'].convert("RGB").resize(target_resolution)

        test_segmentations.append(seg_img)
        test_instances.append(inst_img)

        test_captions.append(item['text'])

    print(f"✅ Selected data loaded and pre-processed: {len(test_segmentations)} frames.")
    return test_segmentations, test_instances, test_captions, original_indices

# --- 5. GENERATION FUNCTION (Modificato per i nomi delle variabili) ---

def generate_image(
    pipe,
    seg_image,
    inst_image,
    prompt,
    model_data,
    prev_image,
    split_temporal,
    control_guidance_start,
    control_guidance_end,
    condscale_temporal, 
    condscale_seg, 
    condscale_inst, 
    guidance: float = 3.0,
    set_seed: bool = True,
    use_guess_mode: bool = True,
) -> Image.Image:


    # control_guidance_start=[start_seg_cond, start_instance_cond, start_temporal_cond],
    #         control_guidance_end=[end_seg_cond, end_instance_cond, end_temporal_cond],
    """
    Generate one frame using 3 ControlNets: [0: Seg, 1: Inst, 2: Temp]
    """
    if set_seed:
        generator = torch.Generator(device=pipe.device).manual_seed(50)
    else:
        generator = None

    

    seg_scale = condscale_seg
    inst_scale = condscale_inst
    temp_scale = condscale_temporal if prev_image is not None else 0.0

    control_image_temp = prev_image if prev_image is not None else seg_image
    control_images = [seg_image, inst_image, control_image_temp]

    controlnet_scales = [seg_scale, inst_scale, temp_scale]
    print(control_guidance_start)
    print(control_guidance_end)
    with torch.no_grad():
        result = pipe(
            prompt,
            image=control_images,
            negative_prompt="blurry, distorted, street without street lines",
            controlnet_conditioning_scale=controlnet_scales,
            height=model_data["size"]["y"],
            width=model_data["size"]["x"],
            num_inference_steps=STABLE_DIFF_STEPS,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            guidance_scale=guidance,
            guess_mode=use_guess_mode,
            output_type="pil",
            generator=generator,
        )

        out_image = result.images[0]

    return out_image

# --- 6. MAIN EXECUTION ---

def main():
    # 1. Download Models (Gestisce la cache)
    model_root = download_models_and_config()

    # 2. Load Pipeline
    pipe, model_data, target_resolution = load_pipeline_models(model_root)
    
    # 3. Determina gli indici da processare
    
    ds_full = load_dataset(TRAINING_DATASET_NAME, split="train")
    total_frames = len(ds_full)
    N = FRAMES_TO_PROCESS_N
    
    # Seleziona solo i primi N frame
    max_index = min(N, total_frames)
    indices_to_process = list(range(max_index))
    total_to_process = len(indices_to_process)

    print("-" * 50)
    print(f"Dataset totale: {total_frames} frames.")
    print(f"Processing ONLY the first {total_to_process} frames (N={N}).")
    print(f"Indici: 0 to {max_index - 1}")
    print("-" * 50)


    # 4. Load and Preprocess Dataset (SOLO I FRAME NECESSARI)
    seg_list, inst_list, caption_list, original_indices = load_and_preprocess_dataset(
        target_resolution, 
        indices_to_process
    )
    # [end_seg_cond, end_instance_cond, end_temporal_cond]
    # [start_seg_cond, start_instance_cond, start_temporal_cond]
    # 5. Execute Generation
    

    scale_values = [0.7, 1.1, 1.5]
    cond_scales = list(product(scale_values, repeat=3))




    # Esempio d'uso:
    final_control_dicts = generate_control_dicts()
    print(f"Totale dizionari di controllo: {len(final_control_dicts)}")
    print("\nEsempi di nuove combinazioni:")
    for d in final_control_dicts:
        print(d)
    output_dir_main = f"./offline_outputs_test"
    os.makedirs(output_dir_main, exist_ok=True)
    for caption in [True,False]:

            condscale_seg = 0.7
            condscale_inst = 0.7
            condscale_temporal = 1.1
            for control_dict in final_control_dicts:
                control_guidance_start=control_dict["start"]
                control_guidance_end=control_dict["end"]

                output_dir = f"./offline_outputs_test/CAPTIONING_{caption}__CONDITIONING_seg_{condscale_seg}_inst_{condscale_inst}_temp_{condscale_temporal}__SPLITSTART_seg_{control_guidance_start[0]}_inst_{control_guidance_start[1]}_temp_{control_guidance_start[2]}__SPLITEND_seg_{control_guidance_end[0]}_inst_{control_guidance_end[1]}_temp_{control_guidance_end[2]}"
                # 2. CONTROLLO: Se la cartella esiste già, salta all'iterazione successiva
                if os.path.exists(output_dir):
                    print(f"Skipping existing directory: {output_dir}")
                    continue

                # 3. Se non esiste, creala e procedi
                os.makedirs(output_dir, exist_ok=True)
                
                prev_generated = None
                print(f"\nStarting generation of {total_to_process} frames...for {output_dir}")
                for i, frame_index in enumerate(original_indices):
                    seg = seg_list[i]
                    inst = inst_list[i]
                    current_prompt = caption_list[i]

                    print(f"Processing [{i+1}/{total_to_process}] (Original Index: {frame_index}): Prompt: \"{current_prompt}\"")

                    # Il condizionamento temporale è basato solo sul frame precedente nella lista processata
                    if i == 0:
                        split_temp_val = 0
                        prev_img = None
                    else:
                        split_temp_val = 30
                        prev_img = prev_generated

                    out_img = generate_image(
                        pipe,
                        seg_image=seg,
                        inst_image=inst,
                        prompt=current_prompt,
                        model_data=model_data,
                        prev_image=prev_img,
                        split_temporal=split_temp_val,
                        control_guidance_start=control_guidance_start,
                        control_guidance_end=control_guidance_end,
                        condscale_seg=condscale_seg,
                        condscale_inst=condscale_inst,
                        condscale_temporal=condscale_temporal,
                        guidance=3.0,
                        set_seed=True,
                        use_guess_mode= not caption,
                    )
                    

                    filename = f"frame_{frame_index:06d}.png"
                    save_path = os.path.join(output_dir, filename)
                    out_img.save(save_path)

                    prev_generated = out_img

                print(f"\nDone! Images saved in {output_dir}")

if __name__ == "__main__":
    main()
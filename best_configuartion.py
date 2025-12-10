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
from typing import List, Tuple, Dict, Any

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

HF_TOKEN = ""
MODEL_REPO = "jannu99/gurickestraemp4_25_12_02_model"
TRAINING_DATASET_NAME = "jannu99/gurickestraemp4_25_12_02"

# Paths
LOCAL_MODEL_DIR = "./gurickestraemp4_25_12_02_local"
OUTPUT_DIR_MAIN = "./best_offline_outputs_test"

# Limita l'elaborazione ai primi N frame
FRAMES_TO_PROCESS_N = 100 

# Pipeline Constants
STABLE_DIFF_STEPS = 50

# --- SCALE FISSE ---
CONDSCALE_SEG = 0.7
CONDSCALE_INST = 0.7
CONDSCALE_TEMPORAL = 1.1

# Definisce il tipo per chiarezza
ControlDict = Dict[str, List[float]]

# Costanti indici
SEG_IDX = 0
INST_IDX = 1
TEMP_IDX = 2
CN_INDICES = [SEG_IDX, INST_IDX, TEMP_IDX]

# --- FUNZIONE GENERAZIONE DIZIONARI ---

def generate_control_dicts(strength_rank: List[int], order_rank: List[int], separation: bool = False) -> List[ControlDict]:
    """
    Crea la lista di dizionari di controllo.
    - Se separation=True: Restituisce i 16 schemi hardcoded di overlapping.
    - Se separation=False: Genera i 7 schemi sequenziali basati su forza e ordine.
    """
    
    if separation: 
        # 16 Schemi di overlapping hardcoded
        control_dicts: List[ControlDict] = [
            {'start': [0.00, 0.11, 0.31], 'end': [0.10, 0.30, 1.00]},
            {'start': [0.00, 0.11, 0.34], 'end': [0.10, 0.34, 1.00]},
            {'start': [0.00, 0.34, 0.00], 'end': [0.33, 1.00, 0.33]},
            {'start': [0.00, 0.34, 0.34], 'end': [0.33, 1.00, 1.00]},
            {'start': [0.34, 0.00, 0.00], 'end': [1.00, 0.33, 0.33]},
            {'start': [0.34, 0.00, 0.34], 'end': [1.00, 0.33, 1.00]},
            {'start': [0.34, 0.34, 0.00], 'end': [1.00, 1.00, 0.33]},
            {'start': [0.34, 0.34, 0.34], 'end': [1.00, 1.00, 1.00]},
            {'start': [0.00, 0.00, 0.00], 'end': [0.66, 0.66, 0.66]},
            {'start': [0.00, 0.00, 0.67], 'end': [0.66, 0.66, 1.00]},
            {'start': [0.00, 0.67, 0.00], 'end': [0.66, 1.00, 0.66]},
            {'start': [0.00, 0.67, 0.67], 'end': [0.66, 1.00, 1.00]},
            {'start': [0.67, 0.00, 0.00], 'end': [1.00, 0.66, 0.66]},
            {'start': [0.67, 0.00, 0.67], 'end': [1.00, 0.66, 1.00]},
            {'start': [0.67, 0.67, 0.00], 'end': [1.00, 1.00, 0.66]},
            {'start': [0.67, 0.67, 0.67], 'end': [1.00, 1.00, 1.00]}
        ]
        return control_dicts

    else: 
        # Generazione Dinamica Sequenziale (7 Schemi)
        # Percentuali di durata da testare
        DURATION_SCHEMES: List[Tuple[float, float, float]] = [
            (0.10, 0.20, 0.70), (0.10, 0.10, 0.80), (0.10, 0.30, 0.60), 
            (0.10, 0.40, 0.50), (0.20, 0.20, 0.60), (0.20, 0.30, 0.50), 
            (0.30, 0.30, 0.40)
        ]
        
        # Ordina gli indici dei CN in base al rank di forza
        strength_sort_order = [i for _, i in sorted(zip(strength_rank, CN_INDICES))]
        
        # Ordina gli indici dei CN in base all'ordine di esecuzione
        order_indices = [i for _, i in sorted(zip(order_rank, CN_INDICES))]
        
        generated_dicts: List[ControlDict] = []
        
        for duration_scheme in DURATION_SCHEMES:
            sorted_durations = sorted(duration_scheme)
            assigned_durations = [0.0, 0.0, 0.0] 
            
            # Assegna la durata in base al rank di forza
            assigned_durations[strength_sort_order[0]] = sorted_durations[0] 
            assigned_durations[strength_sort_order[1]] = sorted_durations[1] 
            assigned_durations[strength_sort_order[2]] = sorted_durations[2] 
            
            starts = [0.0, 0.0, 0.0]
            ends = [0.0, 0.0, 0.0]
            current_time = 0.0
            
            # Costruisce lo schema sequenziale
            for cn_index in order_indices:
                duration = assigned_durations[cn_index]
                starts[cn_index] = round(current_time, 3)
                ends[cn_index] = round(current_time + duration, 3)
                current_time += duration 
            
            generated_dicts.append({'start': starts, 'end': ends})
            
        return generated_dicts


# --- DOWNLOAD E LOAD MODELLI ---

def check_essential_files(model_dir: str) -> bool:
    if not os.path.exists(model_dir): return False
    required_files = ["config.json", "stable_diffusion/pytorch_lora_weights.safetensors", "controlnet_instance/diffusion_pytorch_model.safetensors"]
    for rel_path in required_files:
        if not os.path.exists(os.path.join(model_dir, rel_path)): return False
    return True

def download_models_and_config() -> str:
    if check_essential_files(LOCAL_MODEL_DIR):
        print(f"✅ Modelli essenziali già trovati in {LOCAL_MODEL_DIR}. Saltando il download.")
        return LOCAL_MODEL_DIR
    print("⏳ Logging into Hugging Face..."); login(token=HF_TOKEN, add_to_git_credential=False)
    allow_patterns = ["config.json", "lora_weights/*", "stable_diffusion/*", "controlnet_segmentation/diffusion_pytorch_model.safetensors", "controlnet_segmentation/config.json", "controlnet_instance/diffusion_pytorch_model.safetensors", "controlnet_instance/config.json", "controlnet_tempconsistency/diffusion_pytorch_model.safetensors", "controlnet_tempconsistency/config.json"]
    model_root = snapshot_download(repo_id=MODEL_REPO, repo_type="model", local_dir=LOCAL_MODEL_DIR, local_dir_use_symlinks=False, allow_patterns=allow_patterns)
    return model_root

def load_pipeline_models(model_root: str) -> Tuple[StableDiffusionControlNetPipeline, Dict[str, Any], Tuple[int, int]]:
    config_path = os.path.join(model_root, "config.json");
    with open(config_path, "r") as f: model_data = json.load(f)
    IMG_W, IMG_H = model_data["size"]["x"], model_data["size"]["y"]
    seg_dir_name, temp_dir_name, instance_dir_name = model_data["controlnet_segmentation"], model_data["controlnet_tempconsistency"], model_data["controlnet_instance"]
    controlnet_seg_path, controlnet_temp_path, controlnet_instance_path = os.path.join(model_root, seg_dir_name), os.path.join(model_root, temp_dir_name), os.path.join(model_root, instance_dir_name)
    print("\n⏳ Loading ControlNet Models...")
    controlnet_seg = ControlNetModel.from_pretrained(controlnet_seg_path, torch_dtype=torch.float16)
    controlnet_temp = ControlNetModel.from_pretrained(controlnet_temp_path, torch_dtype=torch.float16)
    controlnet_instance = ControlNetModel.from_pretrained(controlnet_instance_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(model_data["stable_diffusion_model"], controlnet=[controlnet_seg, controlnet_instance, controlnet_temp], torch_dtype=torch.float16, use_safetensors=True).to(device)
    lora_path = os.path.join(model_root, model_data["lora_weights"])
    print(f"⏳ Loading LoRA state dict from: {lora_path}")
    if lora_path.endswith(".safetensors"):
        lora_state_dict = {}; 
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys(): lora_state_dict[key] = f.get_tensor(key)
    else: lora_state_dict = torch.load(lora_path, map_location="cpu")
    pipe.load_lora_weights(lora_state_dict) 
    print("✅ Pipeline Loaded successfully with 3 ControlNets and LoRA.")
    return pipe, model_data, (IMG_W, IMG_H)

def load_and_preprocess_dataset(target_resolution: Tuple[int, int], indices: List[int]) -> Tuple[List[Image.Image], List[Image.Image], List[str], List[int]]:
    print(f"\n⏳ Loading and preprocessing dataset: {TRAINING_DATASET_NAME}")
    ds = load_dataset(TRAINING_DATASET_NAME, split="train")
    ds_selected = ds.select(indices)
    test_segmentations, test_instances, test_captions = [], [], []
    print(f"⏳ Preprocessing {len(ds_selected)} selected examples, resizing to {target_resolution}...")
    for item in tqdm(ds_selected):
        test_segmentations.append(item['segmentation'].convert("RGB").resize(target_resolution))
        test_instances.append(item['instance'].convert("RGB").resize(target_resolution))
        test_captions.append(item['text'])
    print(f"✅ Selected data loaded and pre-processed: {len(test_segmentations)} frames.")
    return test_segmentations, test_instances, test_captions, indices

# --- GENERAZIONE ---

def generate_image(
    pipe, seg_image, inst_image, prompt, model_data, prev_image, split_temporal, 
    control_guidance_start, control_guidance_end, 
    condscale_temporal, condscale_seg, condscale_inst, 
    guidance: float = 3.0, set_seed: bool = True, use_guess_mode: bool = True,
) -> Image.Image:
    
    if set_seed: generator = torch.Generator(device=pipe.device).manual_seed(50)
    else: generator = None

    temp_scale = condscale_temporal if prev_image is not None else 0.0
    controlnet_scales = [condscale_seg, condscale_inst, temp_scale]
    
    control_image_temp = prev_image if prev_image is not None else seg_image
    control_images = [seg_image, inst_image, control_image_temp]

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
    return result.images[0]

# --- MAIN ---

def main():
    model_root = download_models_and_config()
    pipe, model_data, target_resolution = load_pipeline_models(model_root)
    
    ds_full = load_dataset(TRAINING_DATASET_NAME, split="train")
    max_index = min(FRAMES_TO_PROCESS_N, len(ds_full))
    indices_to_process = list(range(max_index))
    
    print("-" * 50)
    print(f"Processing ONLY the first {len(indices_to_process)} frames.")
    print("-" * 50)

    seg_list, inst_list, caption_list, original_indices = load_and_preprocess_dataset(target_resolution, indices_to_process)
    
    # 🌟 IMPOSTA QUI I TUOI PARAMETRI DI TEST 🌟
    
    # OPZIONE A: Test Sequenziale Dinamico (Separation=False)
    # Esempio: Inst(3)>Temp(2)>Seg(1) come forza, ma Inst(1)->Temp(2)->Seg(3) come ordine
    TEST_SEPARATION = False 
    STRENGTH_RANK = [1, 3, 2] # Forza: Seg(1), Inst(3), Temp(2)
    ORDER_RANK = [3, 1, 2]    # Ordine: Seg(3), Inst(1), Temp(2)

    # OPZIONE B: Test Overlapping Hardcoded (Separation=True)
    # TEST_SEPARATION = True
    # STRENGTH_RANK = None
    # ORDER_RANK = None

    # Genera gli schemi in base alla configurazione
    final_control_dicts = generate_control_dicts(strength_rank=STRENGTH_RANK, order_rank=ORDER_RANK, separation=TEST_SEPARATION) 
    
    print("\n" + "="*60)
    print(f"PLANNING TO RUN {len(final_control_dicts) * 2} CONFIGURATIONS")
    print("="*60)
    
    # Stampa di verifica delle combinazioni prima dell'esecuzione
    for caption in [True, False]:
        for d in final_control_dicts:
            start = d["start"]
            end = d["end"]
            print(f"Run: Caption={caption} | Seg: {start[0]}-{end[0]}, Inst: {start[1]}-{end[1]}, Temp: {start[2]}-{end[2]}")
    
    print("="*60 + "\n")

    os.makedirs(OUTPUT_DIR_MAIN, exist_ok=True)
    
    # Scale FISSE
    condscale_seg = CONDSCALE_SEG
    condscale_inst = CONDSCALE_INST
    condscale_temporal = CONDSCALE_TEMPORAL

    # Ciclo solo su Caption SI/NO
    for caption in [True, False]:
        use_caption = caption
        
        # Ciclo sui dizionari di controllo
        for control_dict in final_control_dicts:
            control_guidance_start = control_dict["start"]
            control_guidance_end = control_dict["end"]
            
            # Costruzione nome cartella esplicito come richiesto
            # Format: ...SPLITSTART_seg_X_inst_Y_temp_Z__SPLITEND_seg_A_inst_B_temp_C
            split_start_str = f"seg_{control_guidance_start[0]}_inst_{control_guidance_start[1]}_temp_{control_guidance_start[2]}"
            split_end_str = f"seg_{control_guidance_end[0]}_inst_{control_guidance_end[1]}_temp_{control_guidance_end[2]}"
            
            output_dir_name = f"CAPTIONING_{use_caption}__CONDITIONING_seg_{condscale_seg}_inst_{condscale_inst}_temp_{condscale_temporal}__SPLITSTART_{split_start_str}__SPLITEND_{split_end_str}"
            output_dir = os.path.join(OUTPUT_DIR_MAIN, output_dir_name)
            
            if os.path.exists(output_dir):
                print(f"Skipping existing directory: {output_dir}")
                continue
            
            os.makedirs(output_dir, exist_ok=True)
            
            prev_generated = None
            print(f"\n--- Processing: {output_dir_name} ---")
            
            for i, frame_index in enumerate(original_indices):
                current_prompt = caption_list[i] if use_caption else ""
                
                # Split temporal fittizio per il primo frame (logica ereditata)
                split_temp_val = 0 if i == 0 else 30 
                prev_img = None if i == 0 else prev_generated

                out_img = generate_image(
                    pipe,
                    seg_image=seg_list[i],
                    inst_image=inst_list[i],
                    prompt=current_prompt,
                    model_data=model_data,
                    prev_image=prev_img,
                    split_temporal=split_temp_val,
                    control_guidance_start=control_guidance_start,
                    control_guidance_end=control_guidance_end,
                    condscale_temporal=condscale_temporal,
                    condscale_seg=condscale_seg,
                    condscale_inst=condscale_inst,
                    guidance=3.0,
                    set_seed=True,
                    use_guess_mode=not use_caption,
                )
                
                filename = f"frame_{frame_index:06d}.png"
                out_img.save(os.path.join(output_dir, filename))
                prev_generated = out_img

    print(f"\nDone! All tests completed.")

if __name__ == "__main__":
    main()
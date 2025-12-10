import os
import json
import csv
from PIL import Image
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any, Union

# --- Core Metric Imports ---
import cv2 
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from scipy import linalg
from sklearn.metrics import pairwise_distances
from glob import glob
import pandas as pd
from natsort import natsorted
from shapely.geometry import box
from collections import Counter
# ---------------------------

# ==============================================================================
# --- Configuration Constants ---
# ==============================================================================
TRAINING_DATASET_NAME = "jannu99/gurickestraemp4_25_12_02"
FRAMES_TO_PROCESS_N = 250
TARGET_RESOLUTION = (512, 512)

# FOLDER PATHS
GROUND_TRUTH_CACHE = 'offline_outputs_test_results/a1_GROUND_TRUTH' # Folder A (Real Images)
FOLDER_B_PATH = 'offline_outputs_test_best'  # Root folder for all subfolders (Fake Images)
OUTPUT_DIR_REPORTS = 'offline_outputs_test_results/a2_METRICS' # Folder to save JSON/CSV reports

# PERCEPTUAL/SEMANTIC CONFIG
SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024" 

# VEHICLE CONSISTENCY CONFIG
VEHICLE_CLASSES = {2: "car", 3: "motorbike", 5: "bus", 7: "truck", 1: "bicycle"}
MIN_VEHICLE_AREA = 600
IOU_THRESHOLD = 0.5
YOLO_MODEL_NAME = "yolov8n.pt" 

# ==============================================================================
# --- Utility & Setup Functions ---
# ==============================================================================

def get_device():
    """Returns the torch device available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# --- Ground Truth Loading ---

def load_or_save_ground_truth(
    target_resolution: Tuple[int, int], 
    n_frames: int,
    output_dir: str
) -> List[str]:
    """Loads specified frames from the HF dataset, saves them to a cache, and returns filenames."""
    
    indices_to_process = list(range(n_frames))
    
    if os.path.exists(output_dir):
        saved_files = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if len(saved_files) == n_frames:
            print(f"✅ Ground truth frames already exist in {output_dir} ({len(saved_files)} files). Using cached version.")
            return saved_files
    
    print(f"\n⏳ Loading and saving original frames to Folder A Cache: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        ds = load_dataset(TRAINING_DATASET_NAME, split="train")
    except Exception as e:
        print(f"Error loading dataset {TRAINING_DATASET_NAME}: {e}")
        return []

    ds_selected = ds.select(indices_to_process)
    
    saved_filenames = []
    print(f"⏳ Saving {len(ds_selected)} original frames...")

    for item in tqdm(ds_selected):
        try:
            original_img = item['image'].convert("RGB").resize(target_resolution) 
        except KeyError:
             print("Error: Could not find 'image' key.")
             continue
        
        frame_index = item.get('frame_index', -1)
        
        if frame_index == -1:
             frame_index = len(saved_filenames) # Use sequential index as fallback
        
        filename = f"frame_{frame_index:06d}.png"
        save_path = os.path.join(output_dir, filename)
        original_img.save(save_path)
        saved_filenames.append(filename)

    print(f"✅ All ground truth frames saved to {output_dir}.")
    return saved_filenames

# --- SegFormer Helpers (for CPL/SegScore) ---

def get_segformer_model():
    """Initializes and returns the SegFormer model and its image processor."""
    device = get_device()
    print(f"\n⏳ Initializing SegFormer model on device: {device}...")
    try:
        image_processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)
        model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
        model.eval().to(device)
        print("✅ SegFormer Model Initialized.")
        return model, image_processor
    except Exception as e:
        print(f"Error initializing SegFormer model ({SEGFORMER_MODEL}): {e}")
        print("⚠️ CPL and SegScore metrics will fail.")
        return None, None

def decode_cityscapes_mask(predicted_mask):
    """Decodes the predicted class IDs into Cityscapes color mask."""
    cityscapes_palette = [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
        (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
        (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
        (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ]
    h, w = predicted_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(cityscapes_palette):
        rgb_mask[predicted_mask == class_id] = color
    return rgb_mask

def encode_cityscapes_mask(rgb_img, carla_mode=False):
    """Encodes an RGB Cityscapes-like mask into class IDs (0-18)."""
    rgb = np.array(rgb_img)
    h, w, _ = rgb.shape
    label_mask = np.full((h, w), fill_value=-1, dtype=np.int64)

    segmentation_colors = [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
        (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
        (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
        (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ]

    for class_id, color in enumerate(segmentation_colors):
        mask = np.all(rgb == color, axis=-1)
        label_mask[mask] = class_id

    return torch.from_numpy(label_mask)


# --- YOLO Model Loading (for Vehicle Consistency) ---

class MockYoloResult:
    def __init__(self, data):
        self.boxes = self.MockBoxes(data)
    class MockBoxes:
        def __init__(self, data):
            # data is a tensor/numpy array [x1, y1, x2, y2, conf, cls]
            self.data = torch.tensor(data)
        def cpu(self): return self

def load_yolo_model():
    """Loads the YOLO model (assuming Ultralytics dependency)."""
    try:
        from ultralytics import YOLO
        print(f"\n⏳ Loading YOLO model: {YOLO_MODEL_NAME}...")
        model = YOLO(YOLO_MODEL_NAME)
        print("✅ YOLO Model Loaded.")
        return model
    except ImportError:
        print("🛑 Ultralytics YOLO not installed. Vehicle Consistency will be skipped or use mock data.")
        return None
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def calculate_yolo_image(yolo_model, pil_image: Image.Image):
    """Calculates YOLO results for a PIL image."""
    if yolo_model is None:
        # Return mock results if model failed to load
        return [MockYoloResult([])], 0.0
    
    results = yolo_model(pil_image, verbose=False)
    return results, 0.0

# Transforms for YOLO input (512x512)
image_transforms_512 = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
    ]
)

# --- Vehicle Consistency Core Logic ---

def iou(box1: List[float], box2: List[float]) -> float:
    """Calculates IoU between two bounding boxes."""
    b1 = box(box1[0], box1[1], box1[2], box1[3])
    b2 = box(box2[0], box2[1], box2[2], box2[3])
    return b1.intersection(b2).area / b1.union(b2).area if b1.union(b2).area > 0 else 0

def extract_vehicles(results) -> List[Dict[str, Union[List[float], float, str]]]:
    """Extracts vehicle bounding boxes, confidence, and class name from YOLO results."""
    vehicles = []
    for r in results:
        # r.boxes.data is [x1, y1, x2, y2, conf, cls]
        for data_row in r.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = data_row
            cls = int(cls)
            if cls in VEHICLE_CLASSES:
                vehicles.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "class": VEHICLE_CLASSES[cls]
                })
    return vehicles

def match_vehicles(real_vehicles, gen_vehicles, iou_thresh=IOU_THRESHOLD):
    """Matches real vehicles to generated vehicles based on IoU."""
    matches = []
    used_gen_indices = set()

    for rv in real_vehicles:
        best_match = None
        best_iou = 0
        best_i = -1
        for i, gv in enumerate(gen_vehicles):
            if i in used_gen_indices:
                continue
            iou_score = iou(rv["bbox"], gv["bbox"])
            if iou_score > best_iou:
                best_iou = iou_score
                best_match = (rv, gv, iou_score)
                best_i = i

        if best_match and best_iou >= iou_thresh:
            matches.append(best_match)
            used_gen_indices.add(best_i)

    return matches, used_gen_indices

def filter_large_vehicles(vehicles, min_area=MIN_VEHICLE_AREA):
    """Filters vehicles based on minimum bounding box area."""
    return [v for v in vehicles if (v["bbox"][2]-v["bbox"][0])*(v["bbox"][3]-v["bbox"][1]) >= min_area]


# ==============================================================================
# --- Image-Level Metric Calculation Functions (FIXED & MODIFIED) ---
# ==============================================================================

def compute_ssim(image1_cv2: np.ndarray, image2_cv2: np.ndarray) -> float:
    """Calculates Structural Similarity Index (SSIM)."""
    image1_gray = cv2.cvtColor(image1_cv2, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2_cv2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(image1_gray, image2_gray, data_range=image1_gray.max() - image1_gray.min(), full=True)
    return float(score)

def compute_psnr(image1_cv2: np.ndarray, image2_cv2: np.ndarray) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR)."""
    mse = np.mean((image1_cv2 - image2_cv2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)

def compute_mse(image1_cv2: np.ndarray, image2_cv2: np.ndarray) -> float:
    """Calculates Mean Squared Error (MSE)."""
    return float(np.mean((image1_cv2 - image2_cv2) ** 2))

def compute_cpl(image1_rgb: np.ndarray, image2_rgb: np.ndarray, model_cpl, transform_cpl) -> float:
    """Calculates CLIP-Perceptual Loss (CPL) using a SegFormer model's features."""
    img1_pil = Image.fromarray(image1_rgb)
    img2_pil = Image.fromarray(image2_rgb)
    
    inputs1 = transform_cpl(images=img1_pil, return_tensors="pt")
    inputs2 = transform_cpl(images=img2_pil, return_tensors="pt")

    device = model_cpl.device
    img1_tensor = inputs1["pixel_values"].to(device)
    img2_tensor = inputs2["pixel_values"].to(device)
    
    with torch.no_grad():
        outputs1 = model_cpl(img1_tensor, output_hidden_states=True)
        outputs2 = model_cpl(img2_tensor, output_hidden_states=True)
        
        features1 = outputs1.hidden_states[-1] 
        features2 = outputs2.hidden_states[-1]

    cpl_loss = torch.nn.functional.mse_loss(features1, features2).item()
    return cpl_loss

def calculate_semantic_segmentation_score(model_seg: SegformerForSemanticSegmentation, image_extractor: SegformerImageProcessor, image_seg: np.ndarray, image_created: np.ndarray, carla_mode=False) -> Tuple[float, np.ndarray]:
    """Calculates the MSE of class IDs between predicted and ground truth segmentation."""
    inputs = image_extractor(images=image_created, return_tensors="pt")
    inputs = {k: v.to(model_seg.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_seg(**inputs)
        logits = outputs.logits
        target_size = image_created.shape[:2]

        upsampled = torch.nn.functional.interpolate(
            logits, size=target_size, mode="bilinear", align_corners=False
        )
        predicted = upsampled.argmax(1)[0].cpu().numpy()

    segmentation_image = decode_cityscapes_mask(predicted)

    if image_seg is None:
        return -1.0, segmentation_image 

    pred_ids = encode_cityscapes_mask(segmentation_image, carla_mode=carla_mode)
    gt_ids   = encode_cityscapes_mask(image_seg, carla_mode=carla_mode)

    pred_ids = pred_ids.float()
    gt_ids   = gt_ids.float()

    mse_score = torch.nn.functional.mse_loss(pred_ids, gt_ids).item()
    return mse_score, segmentation_image


def calculate_single_metrics_all(
    image_a_path: str, 
    image_b_path: str,
    segformer_model,
    image_processor,
    yolo_model
) -> dict:
    """
    Calculates all metrics (Pixel, Perceptual, and Vehicle Consistency) 
    for a single pair of images.
    """
    metrics = {
        'MSE': -3, 'PSNR': -3, 'SSIM': -3, 'CPL': -3, 'SegScore': -3,
        'Veh_Recall': -3, 'Veh_Precision': -3, 'Veh_AvgIoU': -3
    }
    
    try:
        # Load images for Pixel/Perceptual (OpenCV/BGR)
        img_a_cv2 = cv2.imread(image_a_path, 1) 
        img_b_cv2 = cv2.imread(image_b_path, 1)

        if img_a_cv2 is None or img_b_cv2 is None:
             return {k: -2 for k in metrics.keys()}

        if img_a_cv2.shape != img_b_cv2.shape:
            return {k: -1 for k in metrics.keys()}

        img_a_rgb = cv2.cvtColor(img_a_cv2, cv2.COLOR_BGR2RGB)
        img_b_rgb = cv2.cvtColor(img_b_cv2, cv2.COLOR_BGR2RGB)

        # 1. Pixel-based Metrics
        metrics['MSE'] = round(compute_mse(img_a_cv2, img_b_cv2), 4)
        metrics['PSNR'] = round(compute_psnr(img_a_cv2, img_b_cv2), 4)
        metrics['SSIM'] = round(compute_ssim(img_a_cv2, img_b_cv2), 4)
        
        # 2. Perceptual/Semantic Metrics
        if segformer_model is not None:
            metrics['CPL'] = round(compute_cpl(img_a_rgb, img_b_rgb, segformer_model, image_processor), 4)
            seg_score_val, _ = calculate_semantic_segmentation_score(
                segformer_model, image_processor, 
                image_seg=img_a_rgb, 
                image_created=img_b_rgb 
            )
            metrics['SegScore'] = round(seg_score_val, 4)
        
        # 3. Vehicle Consistency Metrics
        if yolo_model is not None:
            # Prepare PIL images for YOLO
            real_image_pil = Image.fromarray(img_a_rgb).convert("RGB")
            gen_image_pil = Image.fromarray(img_b_rgb).convert("RGB")
            
            # Apply 512x512 transform for consistent YOLO input resolution
            real_image_t = image_transforms_512(real_image_pil)
            gen_image_t = image_transforms_512(gen_image_pil)

            yolo_results_real, _ = calculate_yolo_image(yolo_model, real_image_t)
            yolo_results_sim, _ = calculate_yolo_image(yolo_model, gen_image_t)

            real_vehicles = extract_vehicles(yolo_results_real)
            gen_vehicles = extract_vehicles(yolo_results_sim)

            real_vehicles = filter_large_vehicles(real_vehicles, min_area=MIN_VEHICLE_AREA)
            gen_vehicles = filter_large_vehicles(gen_vehicles, min_area=MIN_VEHICLE_AREA)

            matches, _ = match_vehicles(real_vehicles, gen_vehicles, iou_thresh=IOU_THRESHOLD)
            
            real_count = len(real_vehicles)
            gen_count = len(gen_vehicles)
            match_count = len(matches)

            # Metrics
            recall = match_count / real_count if real_count else 0
            precision = match_count / gen_count if gen_count else 0
            ious = [m[2] for m in matches]
            avg_iou = np.mean(ious) if ious else 0

            metrics['Veh_Recall'] = round(recall, 4)
            metrics['Veh_Precision'] = round(precision, 4)
            metrics['Veh_AvgIoU'] = round(avg_iou, 4)
        
        return metrics
    
    except Exception as e:
        # If any unexpected error occurs during processing
        # print(f"Critical Error processing images {os.path.basename(image_a_path)} and {os.path.basename(image_b_path)}: {e}")
        return {k: -4 for k in metrics.keys()}


def calculate_distribution_metrics_avg(all_results: dict) -> dict:
    """Calculates standard deviation across all image-level metrics, including vehicle consistency."""
    metrics_to_track = ['MSE', 'PSNR', 'SSIM', 'CPL', 'SegScore', 'Veh_Recall', 'Veh_Precision', 'Veh_AvgIoU']
    distribution_metrics = {}
    
    for metric_name in metrics_to_track:
        values = []
        for metrics in all_results.values():
            val = metrics.get(metric_name)
            # Filter out error codes (< 0) and Infinity (for PSNR)
            if isinstance(val, (int, float)) and val >= 0 and val != float('inf'):
                values.append(val)

        if not values:
            distribution_metrics[f'stdev_{metric_name}'] = 0.0
            continue
        
        stdev = np.std(values)
        distribution_metrics[f'stdev_{metric_name}'] = round(stdev, 4)

    return distribution_metrics

# ==============================================================================
# --- Distribution-Level Metric Calculation Functions (FIXED) ---
# ==============================================================================

image_transforms_inception = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ]
)

class ImageFolderDataset(Dataset):
    def __init__(self, root, exts=(".png",".jpg",".jpeg",".bmp",".webp")):
        self.paths = []
        for e in exts:
            self.paths.extend(glob(os.path.join(root, f"**/*{e}"), recursive=True))
        if not self.paths:
             raise ValueError(f"No images found in {root}")
        self.tf = image_transforms_inception

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            return self.tf(im.convert("RGB"))

class InceptionPool3(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the full Inception V3 model
        net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        net.eval()
        for p in net.parameters(): 
            p.requires_grad = False
            
        # We need to construct the feature extractor by filtering out incompatible layers.
        # Inception V3 structure: ... -> Mixed_6e -> AuxLogits -> Mixed_7a -> ... -> AvgPool -> Dropout -> FC
        layers = []
        for name, module in net.named_children():
            # Skip the auxiliary classifier, dropout, and final linear layer
            if name in ['AuxLogits', 'dropout', 'fc']:
                continue
            layers.append(module)

        # 'layers' now ends with the original 'avgpool'. 
        # We slice it off ([:-1]) to stop at Mixed_7c (feature map size: 2048 x 8 x 8)
        # and explicitly define our own pooling to ensure correct output dimensions.
        self.features = nn.Sequential(*layers[:-1]) 
        
        # Define global average pooling to get a 2048-dim vector
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self,x):
        with torch.no_grad():
            # Pass input through the main feature extraction path (up to Mixed_7c)
            x = self.features(x)
            
            # Apply global pooling: [B, 2048, 8, 8] -> [B, 2048, 1, 1]
            x = self.pool(x)
            
            # Flatten to [B, 2048] for metric calculation
            return torch.flatten(x, 1)

# Helper functions for feature extraction and statistics
def get_features(loader, model, device):
    feats=[]
    for x in tqdm(loader, desc="Extracting Features"):
        feats.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(feats,0)

def get_logits(loader, model, device):
    probs=[]
    with torch.no_grad():
        for x in tqdm(loader, desc="Calculating Logits"):
            logits = model(x.to(device))
            if isinstance(logits,tuple): logits=logits[0]
            probs.append(F.softmax(logits,dim=1).cpu().numpy())
    return np.concatenate(probs,0)

def compute_stats(feats): return feats.mean(0), np.cov(feats,rowvar=False)

def _sqrtm(c1,c2,eps=1e-6):
    c1=c1.copy(); c2=c2.copy()
    c1.flat[::c1.shape[0]+1]+=eps
    c2.flat[::c2.shape[0]+1]+=eps
    cov,info = linalg.sqrtm(c1.dot(c2),disp=False)
    cov = cov.real if np.iscomplexobj(cov) else cov
    return cov

# Distribution Metrics implementations
def inception_score(probs,splits=10):
    N=probs.shape[0]; split=N//splits; scores=[]
    for i in range(splits):
        part=probs[i*split:(i+1)*split]
        py=part.mean(0,keepdims=True)
        kl=part*(np.log(part+1e-10)-np.log(py+1e-10))
        scores.append(np.exp(kl.sum(1).mean()))
    return float(np.mean(scores)), float(np.std(scores))

def fid(mu1,s1,mu2,s2):
    diff=mu1-mu2; cov=_sqrtm(s1,s2)
    return float(diff.dot(diff)+np.trace(s1+s2-2*cov))

def kid_poly(X,Y,deg=3,gamma=None,coef0=1.0,subsets=100,subsize=1000):
    rng=np.random.default_rng(123); n=min(len(X),subsize); m=min(len(Y),subsize)
    if gamma is None: gamma=1.0/X.shape[1]
    vals=[]
    for _ in range(subsets):
        Xs=X[rng.choice(len(X),n,False)]; Ys=Y[rng.choice(len(Y),m,False)]
        Kxx=(gamma*Xs@Xs.T+coef0)**deg; Kyy=(gamma*Ys@Ys.T+coef0)**deg
        Kxy=(gamma*Xs@Ys.T+coef0)**deg
        np.fill_diagonal(Kxx,0); np.fill_diagonal(Kyy,0)
        vals.append(Kxx.sum()/(n*(n-1))+Kyy.sum()/(m*(m-1))-2*Kxy.mean())
    vals=np.array(vals)
    return float(vals.mean()),float(vals.std())

def mmd_rbf(X,Y,sigma='median'):
    Z=np.vstack([X,Y])
    if sigma=='median':
        sample=Z if len(Z)<2000 else Z[np.random.choice(len(Z),2000,False)]
        D=pairwise_distances(sample); sigma=np.median(D[D>0])
    gamma=1/(2*sigma*sigma)
    Kxx=np.exp(-gamma*pairwise_distances(X,X,squared=True))
    Kyy=np.exp(-gamma*pairwise_distances(Y,Y,squared=True))
    Kxy=np.exp(-gamma*pairwise_distances(X,Y,squared=True))
    return float(Kxx.mean()+Kyy.mean()-2*Kxy.mean())

def _knn_radii(X, k=3, metric='euclidean', eps=1e-8):
    """Distance from each point to its k-th nearest neighbor (excluding self)."""
    n = len(X)
    if n < 2: return np.full(n, eps, dtype=np.float64)

    k_eff = min(k, n - 1)
    D = pairwise_distances(X, X, metric=metric)
    radii = np.partition(D, kth=k_eff, axis=1)[:, k_eff]
    radii = np.maximum(radii, eps).astype(np.float64)
    return radii

def precision_recall_density_coverage(real_feats, fake_feats, k=3, metric='euclidean', eps=1e-8):
    """Precision/Recall and Density/Coverage metrics based on KNN distances."""
    r_r = _knn_radii(real_feats, k=k, metric=metric, eps=eps)
    f_r = _knn_radii(fake_feats, k=k, metric=metric, eps=eps)

    # Precision / Density: fake -> real
    D_fr = pairwise_distances(fake_feats, real_feats, metric=metric)
    precision = (D_fr <= r_r).any(axis=1).mean().item()

    k_eff_r = min(k, max(1, len(real_feats) - 1))
    kth_fr = np.partition(D_fr, kth=k_eff_r, axis=1)[:, k_eff_r]
    density = ((D_fr <= kth_fr[:, None]).sum(axis=1) / max(1, k_eff_r)).mean().item()

    # Recall / Coverage: real -> fake
    D_rf = pairwise_distances(real_feats, fake_feats, metric=metric)
    recall = (D_rf <= f_r).any(axis=1).mean().item()

    nearest_rf = D_rf.min(axis=1)
    coverage = (nearest_rf <= r_r).mean().item()

    return float(precision), float(recall), float(density), float(coverage)


def calculate_distribution_metrics_full(real_path: str, fake_path: str, device: torch.device) -> Dict[str, float]:
    """Calculates all distribution-level metrics (IS, FID, KID, MMD, PRDC) for one fake set."""
    batch_size = 64
    num_workers = 2

    try:
        real_ds=ImageFolderDataset(real_path)
        fake_ds=ImageFolderDataset(fake_path)
    except ValueError as e:
        # print(f"Skipping distribution metrics for {os.path.basename(fake_path)}: {e}")
        return {
            "IS_mean": -1, "IS_std": -1, "FID": -1, "KID_mean": -1, 
            "KID_std": -1, "MMD_RBF": -1, "Precision": -1, "Recall": -1, 
            "Density": -1, "Coverage": -1
        }
    
    real_loader=DataLoader(real_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    fake_loader=DataLoader(fake_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    feat_net=InceptionPool3().to(device).eval()
    cls_net=models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()

    real_feats=get_features(real_loader,feat_net,device)
    fake_feats=get_features(fake_loader,feat_net,device)

    # IS
    fake_probs=get_logits(fake_loader,cls_net,device)
    is_mean,is_std=inception_score(fake_probs)

    # FID
    mu_r,s_r=compute_stats(real_feats); mu_f,s_f=compute_stats(fake_feats)
    fid_val=fid(mu_r,s_r,mu_f,s_f)

    # KID
    kid_mean,kid_std=kid_poly(real_feats,fake_feats)

    # MMD-RBF
    mmd_val=mmd_rbf(real_feats,fake_feats)

    # Precision / Recall / Density / Coverage
    prec,rec,dens,cov=precision_recall_density_coverage(real_feats,fake_feats)

    return {
        "IS_mean": is_mean, "IS_std": is_std, "FID": fid_val, 
        "KID_mean": kid_mean, "KID_std": kid_std, "MMD_RBF": mmd_val, 
        "Precision": prec, "Recall": rec, "Density": dens, "Coverage": cov,
    }


# ==============================================================================
# --- Main Execution Logic (Combined & Finalized) ---
# ==============================================================================

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types and float('inf')."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float('inf') if obj == float('inf') else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if obj == float('inf'):
            return "Infinity"
        return json.JSONEncoder.default(self, obj)


def process_folders():
    """
    Main function to run Image-Level (including Vehicle), Distribution-Level metrics, 
    and save reports.
    """
    if not os.path.exists(OUTPUT_DIR_REPORTS):
        os.makedirs(OUTPUT_DIR_REPORTS)
        print(f"Created output directory: {OUTPUT_DIR_REPORTS}")

    device = get_device()
    print(f"Using device for PyTorch: {device}")

    # Initialize models
    segformer_model, image_processor = get_segformer_model()
    yolo_model = load_yolo_model()

    # 1. Download/Load Ground Truth Frames (Folder A)
    original_image_names = load_or_save_ground_truth(TARGET_RESOLUTION, FRAMES_TO_PROCESS_N, GROUND_TRUTH_CACHE)
    
    if not original_image_names:
        print(f"Error: Could not load or find original images. Stopping.")
        return

    real_data_path = GROUND_TRUTH_CACHE
    print("-" * 50)

    # 2. Identify Subfolders (Folder B)
    exclude_list = ["old", "depth", "canny","seg", "baseline"]
    subfolders_b = [
        d for d in os.listdir(FOLDER_B_PATH)
        if os.path.isdir(os.path.join(FOLDER_B_PATH, d))
        and d not in exclude_list
    ]

    distribution_results_list = [] # For Distribution Summary CSV
    
    # Initialize all keys for tracking
    all_metric_keys = ['MSE', 'PSNR', 'SSIM', 'CPL', 'SegScore', 'Veh_Recall', 'Veh_Precision', 'Veh_AvgIoU']


    for subfolder_name in natsorted(subfolders_b):
        subfolder_b_path = os.path.join(FOLDER_B_PATH, subfolder_name)
        print(f"\nProcessing subfolder: **{subfolder_name}**")

        # --- A. Image-Level Metrics Calculation (Including Vehicle) ---
        folder_results = {}
        metric_sums = {key: 0.0 for key in all_metric_keys}
        image_count = 0

        for image_name in tqdm(original_image_names, desc=f"Image-Level ({subfolder_name})"):
            path_a = os.path.join(real_data_path, image_name)
            path_b = os.path.join(subfolder_b_path, image_name)

            if not (os.path.exists(path_b) and os.path.exists(path_a)):
                continue
            
            # calculate_single_metrics_all is modified to take yolo_model
            metrics = calculate_single_metrics_all(
                path_a, path_b, 
                segformer_model, image_processor, yolo_model
            )
            folder_results[image_name] = metrics
            image_count += 1

            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value >= 0 and value != float('inf'):
                    metric_sums[key] = metric_sums.get(key, 0) + value

        # Finalize Image-Level Report
        average_metrics = {k: round(v / image_count, 4) for k, v in metric_sums.items()} if image_count > 0 else {k: 0.0 for k in all_metric_keys}
        distribution_avg_metrics = calculate_distribution_metrics_avg(folder_results)
        
        # --- B. Distribution-Level Metrics Calculation ---
        print(f"Running Distribution-Level Metrics for {subfolder_name}...")
        dist_metrics = calculate_distribution_metrics_full(real_data_path, subfolder_b_path, device)
        
        # --- MODIFICATION: ADD DISTRIBUTION METRICS TO JSON AVERAGE_METRICS ---
        average_metrics['FID'] = round(dist_metrics.get('FID', -1.0), 4)
        average_metrics['KID_mean'] = round(dist_metrics.get('KID_mean', -1.0), 4)
        average_metrics['IS_mean'] = round(dist_metrics.get('IS_mean', -1.0), 4)
        average_metrics['MMD_RBF'] = round(dist_metrics.get('MMD_RBF', -1.0), 4)
        average_metrics['PRDC_Precision'] = round(dist_metrics.get('Precision', -1.0), 4)
        average_metrics['PRDC_Recall'] = round(dist_metrics.get('Recall', -1.0), 4)
        average_metrics['PRDC_Density'] = round(dist_metrics.get('Density', -1.0), 4)
        average_metrics['PRDC_Coverage'] = round(dist_metrics.get('Coverage', -1.0), 4)
        
        # Store for the separate CSV report
        distribution_results_list.append({"Folder-Name": subfolder_name, **dist_metrics})


        image_level_report = {
            'subfolder_name': subfolder_name,
            'total_images_compared': image_count,
            'average_metrics': average_metrics,
            'distribution_stdev_metrics': distribution_avg_metrics,
            'image_metrics': folder_results
        }
        
        # Save JSON Report
        json_output_path = os.path.join(OUTPUT_DIR_REPORTS, f"{subfolder_name}_image_level_report.json")
        with open(json_output_path, 'w') as f:
            json.dump(image_level_report, f, indent=4, cls=NpEncoder)
        print(f"  - Image-Level Report (including all metrics) saved to: {json_output_path}")

        print("-" * 50)


    # 3. Save Final Distribution Summary CSV
    if distribution_results_list:
        df_dist = pd.DataFrame(distribution_results_list)
        csv_path_dist = os.path.join(OUTPUT_DIR_REPORTS, f"distribution_summary.csv")
        df_dist.to_csv(csv_path_dist, index=False)
        print(f"✅ Final Distribution Summary saved to {csv_path_dist}")

    print("✅ All processing complete.")


if __name__ == '__main__':
    # Ensure all dependencies are installed:
    # pip install Pillow numpy datasets tqdm opencv-python scikit-image torch torchvision transformers scipy scikit-learn pandas ultralytics
    
    try:
        import torch
        import pandas as pd
        
        print("Starting combined metric analysis process (Image-Level, Distribution, and Vehicle Consistency)...")
        process_folders()
    except ImportError as e:
        print(f"\n🛑 Dependency Error: {e}")
        print("Please ensure you have all required libraries installed, including 'ultralytics' for YOLO and all dependencies from previous steps.")
import os
from huggingface_hub import login
from dotenv import load_dotenv
from datasets import Dataset

from config import DATASETS_FOLDER_NAME
from utils.argparser import parse_dataset_args
from utils.dataset import (
    get_video_dataset, 
    describe_images, 
    get_dataset_features, 
    create_segmentation_data, 
    create_depth_data, 
    create_canny_data
)
from utils.save_data import (
    create_dataset_folders, 
    get_dataset_folder_name, 
    save_dataset, 
    create_dotenv, 
    delete_image_files
)

# 1. Setup Arguments & Environment
args = parse_dataset_args()

create_dotenv()
load_dotenv()
huggingface_token = os.getenv("HF_TOKEN")
if not huggingface_token:
    raise ValueError("⚠️ Huggingface Token not found. Please set the HF_TOKEN in your .env file.")

login(token=huggingface_token)

# 2. Setup Paths
dataset_name = args.name if args.name else get_dataset_folder_name(args.video)
dataset_path = os.path.join(DATASETS_FOLDER_NAME, dataset_name)

create_dataset_folders(dataset_path)

# 3. Process Data
# Gets frames from video (skips if frames already exist in folder)
dataset = get_video_dataset(dataset_path, args.video, args.canny, args.blur, override=args.override)

# Run Segmentation
print("🎨 Generating Segmentation Masks...")
create_segmentation_data(dataset, dataset_path)

# Run Canny (Optional)
if args.canny:
    print("✏️ Generating Canny Edges...")
    create_canny_data(dataset, dataset_path)

# --- [DISABLED] Depth Estimation ---
# print("📏 Generating Depth Maps...")
# create_depth_data(dataset, dataset_path) 

# --- [ENABLED] Image Captioning ---
print("📝 Generating Image Captions...")
describe_images(dataset)

# 4. Create HuggingFace Dataset
# We set features=None because we removed Depth, so the original fixed schema would fail.
# None allows HF to auto-detect the available columns (image, segmentation, text, etc.)
# dataset_features = get_dataset_features(args.canny) 
hf_dataset = Dataset.from_list(dataset, features=None)

# 5. Save & Upload
save_dataset(dataset_path, hf_dataset)
# delete_image_files(dataset_path) # Uncomment to save space after processing
print("\n📂 Dataset saved successfully to:", dataset_path)

if args.upload:
    print(f"\n☁️ Uploading to HuggingFace as: {dataset_name}...")
    hf_dataset.push_to_hub(dataset_name, private=True, token=huggingface_token)
    print("\n💻 Dataset privately uploaded to Huggingface successfully!")
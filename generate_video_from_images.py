import cv2
import os

# --- Configuration ---
main_folder = '/home/davide/Downloads/offline_outputs_test(1)/content'
output_folder = 'try_pipeline' # This will be created in the current directory
fps = 15

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of all items in the directory and filter for folders only
subfolders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

print(f"Found {len(subfolders)} folders to process...")

for folder_name in subfolders:
    image_folder_path = os.path.join(main_folder, folder_name)
    video_save_path = os.path.join(output_folder, f"{folder_name}.mp4")
    
    # 1. Get images
    images = [img for img in os.listdir(image_folder_path) if img.endswith(".png") or img.endswith(".jpg")]
    
    # Skip empty folders to prevent crashes
    if not images:
        print(f"Skipping {folder_name}: No images found.")
        continue

    # Sort them naturally based on digits found in filename
    try:
        images.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    except ValueError:
        print(f"Warning: Filenames in {folder_name} don't contain numbers. Sorting alphabetically.")
        images.sort()

    # 2. Read the first image to get dimensions
    first_image_path = os.path.join(image_folder_path, images[0])
    frame = cv2.imread(first_image_path)
    
    if frame is None:
        print(f"Error reading first frame in {folder_name}. Skipping.")
        continue
        
    height, width, layers = frame.shape

    # 3. Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))

    # 4. Loop through images and write them to video
    print(f"Processing: {folder_name}...")
    for image in images:
        img_path = os.path.join(image_folder_path, image)
        frame = cv2.imread(img_path)
        
        # Resize check: OpenCV requires all frames to be same size. 
        # If your pipeline outputs vary slightly, uncomment the line below:
        # frame = cv2.resize(frame, (width, height))
        
        video.write(frame)

    video.release()

cv2.destroyAllWindows()
print("All videos have been processed and saved.")
import os
import cv2
import csv
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, clips_array

# --- Configuration ---
input_folder = 'videos'          # Change to your new folder path if needed
output_file = 'comparison_final_4x7.mp4'
csv_output_file = 'video_mapping.csv'

# Grid Layout: 28 videos -> 4 rows of 7
cols_per_row = 7
target_total_width = 1920 
margin = 2 
frames_to_keep = 250 

# ==========================================
# STEP 0: FILE CHECKING & CSV EXPORT
# ==========================================
# Get all MP4 files
raw_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
# Sort strictly so the order is deterministic
raw_files.sort()

print(f"Found {len(raw_files)} videos.")
if len(raw_files) == 0:
    raise ValueError("No videos found! Check your input folder.")

# Prepare list for processing and data for CSV
video_files = []
csv_data = []

print(f"Generating CSV mapping: {csv_output_file}...")

with open(csv_output_file, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Header
    writer.writerow(['Grid Number', 'Video Filename'])
    
    for idx, filename in enumerate(raw_files, start=1):
        video_files.append(filename)
        # Add row to CSV: Number -> Filename
        writer.writerow([idx, filename])
        csv_data.append((idx, filename))

print("CSV saved.")

# ==========================================
# STEP 1: LOAD AND PROCESS
# ==========================================
print("Loading videos...")

# Calculate dimensions based on first video
first_video_path = os.path.join(input_folder, video_files[0])
temp_clip = VideoFileClip(first_video_path)
clip_aspect = temp_clip.w / temp_clip.h

# Calculate new width/height to fit 7 cols in 1920px
new_width = int(target_total_width / cols_per_row) - (margin * 2)
new_height = int(new_width / clip_aspect)
temp_clip.close()

def add_text_opencv(get_frame, t, idx):
    """Draws the yellow number on the frame using OpenCV"""
    frame = get_frame(t) 
    frame = frame.copy() # Make writable
    
    text = str(idx)
    font = cv2.FONT_HERSHEY_TRIPLEX # Professional font
    font_scale = 1.0 
    thickness = 2 
    color = (255, 255, 0) # Yellow
    
    # Position: Left Top Corner (x=10, y=30)
    cv2.putText(frame, text, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame

processed_clips = []

for idx, filename in enumerate(video_files, start=1):
    path = os.path.join(input_folder, filename)
    clip = VideoFileClip(path)
    
    # A. Trim
    if clip.fps: 
        max_duration = frames_to_keep / clip.fps
        if clip.duration > max_duration:
            clip = clip.subclip(0, max_duration)
    
    # B. Resize 
    clip = clip.resize(width=new_width)
    
    # C. Add Number 
    # capturing 'current_i=idx' avoids the loop variable bug
    clip = clip.fl(lambda gf, t, current_i=idx: add_text_opencv(gf, t, current_i))
    
    # D. Add Margin
    clip = clip.margin(margin, color=(0,0,0))
    
    processed_clips.append(clip)

# ==========================================
# STEP 2: BUILD GRID (4 Rows of 7)
# ==========================================
grid_rows = []
# We iterate 4 times for 4 rows
for i in range(4):
    start = i * cols_per_row
    end = (i + 1) * cols_per_row
    row_clips = processed_clips[start:end]
    
    # Fill empty spots if less than 7 in a row
    while len(row_clips) < cols_per_row:
        # Create black placeholder
        if processed_clips:
            dur = processed_clips[0].duration
        else:
            dur = 5 # Fallback
            
        black_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        base = ImageClip(black_frame).set_duration(dur)
        row_clips.append(base.margin(margin, color=(0,0,0)))
        
    grid_rows.append(row_clips)

# ==========================================
# STEP 3: RENDER
# ==========================================
print("Composing grid...")
final_video = clips_array(grid_rows, bg_color=(0,0,0))

print(f"Writing video (Resolution: {final_video.w}x{final_video.h})...")
final_video.write_videofile(output_file, codec='libx264', preset='fast', fps=30)
print("Done!")
import cv2
import os
from pathlib import Path


# frame configurations
INPUT_VIDEOS_DIR = "./notebooks/raw_data/videos"# Folder path containing your video files
OUTPUT_FRAMES_DIR = "./notebooks/raw_data/raw_frames" # Folder where extracted image frames will be saved
FRAME_INTERVAL = 300  # Frame sampling rate (e.g. 30 = 1 frame per sec for 30 FPS video)
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.MOV', '.MP4') # Supported formats



def extract_frames_from_directory(input_dir, output_dir, frame_interval):
    """
    Scans the input directory for video files and extracts image frames at a specified interval.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 1. Verify if the input directory exists
    if not input_path.exists():
        print(f"ERROR: Input directory '{input_dir}' does not exist!")
        print(f"Please create the folder '{input_dir}' and place your video files inside.")
        return

    # 2. Search for video files matching supported extensions
    video_files = [f for f in input_path.iterdir() if f.suffix in VIDEO_EXTENSIONS]
    
    if not video_files:
        print(f"No video files (.mp4, .avi, .mkv, .mov) found in '{input_dir}'!")
        return

    # 3. Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(video_files)} video file(s) in '{input_dir}'")
    print(f"Output Directory: '{output_dir}'")
    print(f"Sampling Rate: Saving 1 frame every {frame_interval} frame(s)\n")

    total_saved_all = 0

    # 4. Iterate through each video file and extract frames
    for vid_idx, video_path in enumerate(video_files, start=1):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Could not open video file: {video_path.name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[+] Processing Video [{vid_idx}/{len(video_files)}]: {video_path.name}")
        print(f"    FPS: {fps:.2f} | Total Frames: {total_frames}")

        frame_count = 0
        saved_count = 0
        video_name = video_path.stem

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Reached end of video file

            # Save frame if it hits the specified frame interval
            if frame_count % frame_interval == 0:
                frame_filename = output_path / f"{video_name}_frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                saved_count += 1
                total_saved_all += 1

            frame_count += 1

        cap.release()  # Release video capture resource
        print(f"Saved {saved_count} frame(s) from '{video_path.name}'.\n")

    print(f"Extraction complete! Total {total_saved_all} frame(s) saved in '{output_dir}'.")


if __name__ == "__main__":
    # Execute extraction using configuration variables defined above
    extract_frames_from_directory(
        input_dir=INPUT_VIDEOS_DIR,
        output_dir=OUTPUT_FRAMES_DIR,
        frame_interval=FRAME_INTERVAL
    )
import cv2
import os

def process_buckets(bucket_range, input_dir, output_dir, n, crop_box):
    """
    bucket_range: list of ints, e.g., [1, 2, 3, 4, 5]
    crop_box: tuple (y1, y2, x1, x2) in pixels
    """
    y1, y2, x1, x2 = crop_box
    new_width = x2 - x1
    new_height = y2 - y1

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for b in bucket_range:
        input_path = os.path.join(input_dir, f"bucket{b}.mp4")
        output_path = os.path.join(output_dir, f"bucket{b}_speed_cropped.mp4")

        if not os.path.exists(input_path):
            print(f"Skipping: {input_path} not found.")
            continue

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Use mp4v for compatibility or avc1 for better compression
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

        count = 0
        saved_count = 0

        print(f"Processing Bucket {b}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Keep every Nth frame
            if count % n == 0:
                # CROP LOGIC: [y_start:y_end, x_start:x_end]
                cropped_frame = frame[y1:y2, x1:x2]
                out.write(cropped_frame)
                saved_count += 1
                
            count += 1

        cap.release()
        out.release()
        print(f"Bucket {b} Finished: Kept {saved_count}/{count} frames.")

# --- SETTINGS ---
# Define your crop box here (y_top, y_bottom, x_left, x_right)
# Tip: Use a screenshot of your video in Paint/Preview to find these pixel values
MY_CROP = (200, 1970, 550, 2400) 

process_buckets(
    bucket_range=range(1, 2), # Buckets 1 through 5
    input_dir="videos",
    output_dir="videos/processed",
    n=14, 
    crop_box=MY_CROP
)
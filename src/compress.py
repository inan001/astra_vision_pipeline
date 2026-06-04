import cv2
import os
import glob

# Set up paths relative to your astra_vision_pipeline folder
input_folder = 'data/raw'
output_folder = 'data/resized'

# Create the output folder if it doesn't exist somehow
os.makedirs(output_folder, exist_ok=True)

# Find all JPGs in the raw folder
image_files = glob.glob(os.path.join(input_folder, '*.jpg'))

if not image_files:
    print(f"No JPG images found in {input_folder}. Check your file extensions!")

for img_path in image_files:
    # Read the image
    img = cv2.imread(img_path)
    
    if img is not None:
        filename = os.path.basename(img_path)
        
        # Scale the image down by 50% (maintains aspect ratio)
        height, width = img.shape[:2]
        new_width = int(width * 0.5)
        new_height = int(height * 0.5)
        
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Save to the 'resized' folder with 80% JPEG quality to crush the file size
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        
        # Get the new file size in MB for the console
        new_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Compressed {filename} -> {new_size_mb:.2f} MB")
    else:
        print(f"Failed to read {img_path}")

print("\nDone! Upload the images from 'data/resized' to Roboflow.")
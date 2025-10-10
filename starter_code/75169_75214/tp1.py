###############################################################################
## Computer Vision 2025-2026 - NOVA FCT
## Assignment 1
##
## Student 1 Luca Davì
## Student 2 Marta Negri
##
###############################################################################

import cv2 as cv
import numpy as np
import os
import glob


# Example usage:
if __name__ == "__main__":
    print("Assignment 1")
    #TODO


#Resize every image in the input folder maintaining the aspect-ratio of the images. The
#smaller side of each image should be 512 pixels using cv2. I have an output and imput directory and all the images are
#in jpg format.
def resize_images(input_dir, output_dir, size=512):
   
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files (jpg)
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))

    if not image_paths:
        print("No images found in input folder:", input_dir)
        return

    for img_path in image_paths:
        # Load image
        img = cv.imread(img_path)
       
        # Get original dimensions
        h, w = img.shape[:2]
       
        # Calculate the scaling factor to maintain aspect ratio
        if h < w:
            scale = size / h
        else:
            scale = size / w
       
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
       
        # Resize the image
        img_resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
       
        # Save output
        base = os.path.basename(img_path)
        out_path = os.path.join(output_dir, base)
        cv.imwrite(out_path, img_resized)
        print(f"Saved {out_path}")

# now we use the function
resize_images("input", "output", size=512)



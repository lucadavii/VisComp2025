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
import matplotlib.pyplot as plt
import json

#Resize every image in the input folder maintaining the aspect-ratio of the images. The
#smaller side of each image should be 512 pixels using cv2. I have an output and imput directory and all the images are
#in jpg format.
def resize_images(input_dir, output_dir, size=512):
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    # Get all image files (jpg)
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
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


def create_histogram_json(img_rgb_path):
    # Convert BGR to RGB
    img_rgb = cv.imread(img_rgb_path)
    img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)

    # Split the channels
    R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

    hist_r = np.bincount(R.ravel(), minlength=256)  #ravel converts R matrix to a flat array
    hist_g = np.bincount(G.ravel(), minlength=256)  #bincount counts every ocurrence of every result.
    hist_b = np.bincount(B.ravel(), minlength=256)

    #normalize histograms
    hist_r = hist_r.astype(np.float32)
    hist_g = hist_g.astype(np.float32)
    hist_b = hist_b.astype(np.float32)
    #normalize to avoid exposure differences
    hist_r /= hist_r.sum()
    hist_g /= hist_g.sum()
    hist_b /= hist_b.sum()



    # Compute histograms using cv2
    # hist = cv.calcHist([img_rgb], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    # hist = cv.normalize(hist, hist)

    # Split the histogram into its respective channels
    # hist_r = hist[:, 0, 0]
    # hist_g = hist[0, :, 0]
    # hist_b = hist[0, 0, :]
    #save histogram as json
    hist_data = {
        "red": hist_r.tolist(),
        "green": hist_g.tolist(),
        "blue": hist_b.tolist()
    }
    #print(hist.shape)
    # Create output directory for histograms and save the histogram with the same name as the image
    json_dir = os.path.join(os.path.dirname(img_rgb_path), "histograms")
    os.makedirs(json_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_rgb_path))[0]
    json_path = os.path.join(json_dir, f"{base_name}_histogram.json")

    with open(json_path, "w") as json_file:
        json.dump(hist_data, json_file, indent=4)


def create_histograms(input_dir):
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    if not image_paths:
        print("No images found in input folder:", input_dir)
        return
    for img_path in image_paths:
        create_histogram_json(img_path)
        print(f"Histogram saved for {img_path}")

def histogram_distance(histA, histB):
    
    # Add a small epsilon to avoid division by zero, formula equivalent to the one in cv2
    epsilon = 1e-10
    chi_sq_r = np.sum(((histA[0] - histB[0]) ** 2) / (histA[0] + histB[0] + epsilon))
    chi_sq_g = np.sum(((histA[1] - histB[1]) ** 2) / (histA[1] + histB[1] + epsilon))
    chi_sq_b = np.sum(((histA[2] - histB[2]) ** 2) / (histA[2] + histB[2] + epsilon))
    print(f"Chi-Square distances - R: {chi_sq_r}, G: {chi_sq_g}, B: {chi_sq_b}")
    return np.mean(np.array([chi_sq_r, chi_sq_g, chi_sq_b])) #same thing as computing the chi-squared distance on the combined histogram

def bhattacharyya_distance(histA, histB):
    # Compute the Bhattacharyya distance for each channel
    dist_r = cv.compareHist(histA[0].astype(np.float32), histB[0].astype(np.float32), cv.HISTCMP_BHATTACHARYYA)
    dist_g = cv.compareHist(histA[1].astype(np.float32), histB[1].astype(np.float32), cv.HISTCMP_BHATTACHARYYA)
    dist_b = cv.compareHist(histA[2].astype(np.float32), histB[2].astype(np.float32), cv.HISTCMP_BHATTACHARYYA)
    print(f"Bhattacharyya distances - R: {dist_r}, G: {dist_g}, B: {dist_b}")
    return np.mean(np.array([dist_r, dist_g, dist_b]))

def plot_histogram(hist, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.plot(hist[0], color='r')
    plt.plot(hist[1], color='g')
    plt.plot(hist[2], color='b')
    plt.show()
if __name__ == "__main__":
    resize_images("./input", "./output", size=512)
    create_histograms("./output")
    # Example of computing Chi-Square distance between two histograms
    start_val, end_val = 0, 255  # Exclude first and last bins as they contain possibly over/under-exposed pixels
    with open("./output/histograms/100000_histogram.json", "r") as f:
        histA = json.load(f)
        histA = np.array([histA["red"][start_val:end_val], histA["green"][start_val:end_val], histA["blue"][start_val:end_val]],dtype=np.float32) #First index indicates the channel
        #print(histA.shape)
    with open("./output/histograms/100001_histogram.json", "r") as f:
        histB = json.load(f)
        histB = np.array([histB["red"][start_val:end_val], histB["green"][start_val:end_val], histB["blue"][start_val:end_val]],dtype=np.float32)  # Combine channels for distance calculation
    distance = histogram_distance(histA, histB)
    b_dist=bhattacharyya_distance(histA, histB)

    #as a test, calculate distance between first histogram and all histograms in the folder
    image_paths = sorted(glob.glob(os.path.join("./output/histograms/", "*_histogram.json")))
    #print(image_paths)
    #np array to store distances
    distances = np.zeros((len(image_paths), 2)) #2 distances for each image
    for i, img_path in enumerate(image_paths):
        with open(img_path, "r") as f:
            hist = json.load(f)
            hist = np.array([hist["red"][start_val:end_val], hist["green"][start_val:end_val], hist["blue"][start_val:end_val]],dtype=np.float32) #First index indicates the channel
            distances[i, 0] = histogram_distance(histA, hist)
            distances[i, 1] = bhattacharyya_distance(histA, hist)
    #save distances as csv
    np.savetxt("distances.csv", distances, delimiter=",", header="Chi-Square,Bhattacharyya", comments='')
    #plot distances in two different plots in the same figure
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("Chi-Square Distances from Image 100000")
    plt.xlabel("Image Index")
    plt.ylabel("Chi-Square Distance")
    plt.plot(distances[:, 0], marker='o')
    plt.subplot(2, 1, 2)
    plt.title("Bhattacharyya Distances from Image 100000")
    plt.xlabel("Image Index")
    plt.ylabel("Bhattacharyya Distance")
    plt.plot(distances[:, 1], marker='o', color='orange')
    plt.tight_layout()
    plt.show()

    plot_histogram(histA, "Histogram of Image 100000")
    plot_histogram(histB, "Histogram of Image 100001")
    # print(f"Chi-Square distance between histograms: {distance}")
    # print(f"Bhattacharyya distance between histograms: {b_dist}")
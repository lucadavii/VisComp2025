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
from myMOPS import myMOPS
################################
# Assuming input and output folder paths as sister directories of the current working directory
INPUT_FOLDER_PATH = "../input"
OUTPUT_FOLDER_PATH = "../output"
################################

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
    #print(f"Chi-Square distances - R: {chi_sq_r}, G: {chi_sq_g}, B: {chi_sq_b}")
    return np.mean(np.array([chi_sq_r, chi_sq_g, chi_sq_b])) #same thing as computing the chi-squared distance on the combined histogram

def bhattacharyya_distance(histA, histB):
    # Compute the Bhattacharyya distance for each channel, used just as test and never actually called
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

def group_by_similarity(input_dir, threshold):

    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    #take the first image and put following images in the same folder if their distance is below the threshold,
    #otherwise create a new folder and take the image above the threshold as new reference
    if not image_paths:
        print("No images found in input folder:", input_dir)
        return
    group_index = 0
    reference_hist = None
    target_dir = os.path.join(input_dir, f"similar-{group_index}")
    os.makedirs(target_dir, exist_ok=True) #create first group folder
    for i, img_path in enumerate(image_paths):
        with open(os.path.join(input_dir, "histograms", f"{os.path.splitext(os.path.basename(img_path))[0]}_histogram.json"), "r") as f:
            hist = json.load(f)
            hist = np.array([hist["red"], hist["green"], hist["blue"]],dtype=np.float32) #First index indicates the channel
        if i == 0: #first image is the reference, put it in the first similarity folder and go to next image
            reference_hist = hist
            #move image to target dir
            dst_path = os.path.join(target_dir, os.path.basename(img_path))
            os.replace(img_path, dst_path)
            continue
        distance = histogram_distance(reference_hist, hist)
        if distance < threshold:
            #move image to similarity dir
            dst_path = os.path.join(target_dir, os.path.basename(img_path))
            os.replace(img_path, dst_path)
        else:
            #create new similarity folder and update reference histogram
            group_index += 1
            target_dir = os.path.join(input_dir, f"similar-{group_index}")
            os.makedirs(target_dir, exist_ok=True)
            #move image to new target dir
            dst_path = os.path.join(target_dir, os.path.basename(img_path))
            os.replace(img_path, dst_path)
            reference_hist = hist

def common_histogram_and_white_balance(input_dir):
    similarity_folders = sorted(glob.glob(os.path.join(input_dir, "similar-*")))
    for folder in similarity_folders:
        image_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        if not image_paths:
            print("No images found in folder:", folder)
            continue
        #compute common histogram
        common_hist = np.zeros((3, 256), dtype=np.float32)
        for img_path in image_paths:
            with open(os.path.join(input_dir, "histograms", f"{os.path.splitext(os.path.basename(img_path))[0]}_histogram.json"), "r") as f:
                hist = json.load(f)
                hist = np.array([hist["red"], hist["green"], hist["blue"]],dtype=np.float32) #First index indicates the channel
                common_hist += hist
        common_hist /= len(image_paths)  #average histogram
        print(f"Common histogram computed for folder {folder}")
        with open(os.path.join(folder, "common_histogram.json"), "w") as json_file:
            hist_data = {
                "red": common_hist[0].tolist(),
                "green": common_hist[1].tolist(),
                "blue": common_hist[2].tolist()
            }
            json.dump(hist_data, json_file, indent=4)

        #Extra points if you print the histograms of all images together with the average histogram with labels in a single image figure.
        #for each histogram, plot it in a subfigure and save the figure as an image
        num_images = len(image_paths)
        cols = 3
        total_plots = num_images + 1  # include common histogram
        rows = (total_plots + cols - 1) // cols
        plt.figure(figsize=(15, 5 * rows))
        for i, img_path in enumerate(image_paths):
            with open(os.path.join(input_dir, "histograms", f"{os.path.splitext(os.path.basename(img_path))[0]}_histogram.json"), "r") as f:
                hist = json.load(f)
                hist = np.array([hist["red"], hist["green"], hist["blue"]],dtype=np.float32) #First index indicates the channel
            plt.subplot(rows, cols, i + 1)
            plt.title(f"Histogram of {os.path.basename(img_path)}")
            plt.xlabel("Pixel value")
            plt.ylabel("Frequency")
            plt.xlim([0, 256])
            plt.plot(hist[0], color='r')
            plt.plot(hist[1], color='g')
            plt.plot(hist[2], color='b')
        #plot common histogram in the next available subplot
        plt.subplot(rows, cols, num_images + 1)
        plt.title("Common Histogram")
        plt.xlabel("Pixel value")
        plt.ylabel("Frequency")
        plt.xlim([0, 256])
        plt.plot(common_hist[0], color='r')
        plt.plot(common_hist[1], color='g')
        plt.plot(common_hist[2], color='b')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "histograms_comparison.png"))
        plt.close()
        print(f"Histogram comparison figure saved for folder {folder}")

        #apply histogram equalization to each image based on the common histogram

        avg_rgb = np.zeros(3, dtype=np.float32)
        for c in range(3):
            #compute average pixel value for each channel from common histogram
            avg_rgb[c] = np.sum(np.arange(256) * common_hist[c]) / np.sum(common_hist[c])
        avg_grey = np.mean(avg_rgb)

        for img_path in image_paths:
            img = cv.imread(img_path)
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            #apply white balance
            img_wb = img_rgb.astype(np.float32)
            for c in range(3):
                img_wb[:,:,c] = img_wb[:,:,c] * (avg_grey / avg_rgb[c])
            #clip values to [0,255] and convert back to uint8
            img_wb = np.clip(img_wb, 0, 255).astype(np.uint8)
            #save white balanced image
            out_path = os.path.join(folder, f"wb_{os.path.basename(img_path)}")
            img_wb_bgr = cv.cvtColor(img_wb, cv.COLOR_RGB2BGR)
            cv.imwrite(out_path, img_wb_bgr)
            print(f"White balanced image saved for {img_path}")
def match_images_MOPS(img1, img2):
    myMOPS_instance = myMOPS()
    matched_img=myMOPS_instance.my_draw_matches(img1, img2)
    return matched_img
def match_images_SIFT(img1, img2):
    # Initialize SIFT detector
    sift = cv.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    #keep first 300 matches only
    if len(good_matches) > 500:
        good_matches = good_matches[:500]
    # Draw matches
    matched_image = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None)
    return matched_image
# for every similar folder, compare each image with the first image of the folder and
# verify if it is identical. If it is, save the keypoint comparison in a file named
# `equal-0.jpg, equal-1.jpg, etc' in the same folder.
def check_identical_images(input_dir):
    similarity_folders = sorted(glob.glob(os.path.join(input_dir, "similar-*")))
    for folder in similarity_folders:
        #use only white-balanced images for identical check
        image_paths = sorted(glob.glob(os.path.join(folder, "wb_*.jpg")))
        if not image_paths:
            print("No images found in folder:", folder)
            continue
        reference_img = cv.imread(image_paths[0])
        #compare with other images using sift
        sift = cv.SIFT_create()
        gray_ref = cv.cvtColor(reference_img, cv.COLOR_BGR2GRAY)
        keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_ref, None)
        equal_index = 0
        for img_path in image_paths[1:]:
            img = cv.imread(img_path)
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            keypoints_img, descriptors_img = sift.detectAndCompute(gray_img, None)

            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(descriptors_ref, descriptors_img, k=2)

            # Store all the good matches as per Lowe's ratio test.
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            # If number of good matches is above a threshold, consider images identical,
            if len(good_matches) > 500:
                print(f"Images {os.path.basename(image_paths[0])} and {os.path.basename(img_path)} are identical with {len(good_matches)} good matches.")
                matched_image = cv.drawMatches(reference_img, keypoints_ref, img, keypoints_img, good_matches, None)
                out_path = os.path.join(folder, f"equal-{equal_index}.jpg")
                cv.imwrite(out_path, matched_image)
                equal_index += 1
def print_statistics(dir):
    #For each similar folder print the following output:
    # similar-{a} number of images: {b} ground-truth: {c} precision: {d} averagecolor: {e}
    # * {a} is the sequential number of similar.
    # * {b} is the number of images in that folder.
    # * {c} is the number of correct matches according to the `groundtruth.json`.
    # * {d} is the precision with 3 decimal points $precision=1-(abs(b-c)/c)$
    # * {e} is the RGB value of the average color of that folder(before white
    # balance)
    # • in the end, there should be a global count for all the folders
    # TOTAL number of images: {b} ground-truth: {c} precision: {d}
    similarity_folders = sorted(glob.glob(os.path.join(dir, "similar-*")))
    total_images = 0
    total_groundtruth = 0

    with open(os.path.join(INPUT_FOLDER_PATH, "groundtruth.json"), "r") as f:
        groundtruth_data = json.load(f)


    for folder in similarity_folders:
        #consider images before white balance for counting
        unbalanced_image_paths = [p for p in sorted(glob.glob(os.path.join(folder, "*.jpg"))) if len(os.path.basename(p)) == 10] 
        #count matches: first image is query field, following images are in similar field
        folder_idx = os.path.basename(folder).split("-")[1]
        query_img_name = os.path.basename(unbalanced_image_paths[0])
        similar_imgs_names = [os.path.basename(p) for p in unbalanced_image_paths[1:]]

        #correct matches according to groundtruth
        good_matches=0
        if folder_idx in groundtruth_data.keys() and query_img_name == groundtruth_data[folder_idx]["query"]:
            good_matches += 1 #count query image as good match
            groundtruth_similar = groundtruth_data[folder_idx]["similar"]
            for img_name in similar_imgs_names:
                if img_name in groundtruth_similar:
                    good_matches += 1
        #b is number of images in that folder
        num_images = len(unbalanced_image_paths)  #include query image
        #c is good_matches, counting also query image
        
        if good_matches == 0:
            precision = 0.0
        else:        
            precision = 1 - (np.abs(num_images - good_matches) / (good_matches + 1e-10))

        #cumulative totals
        total_images += num_images
        total_groundtruth += good_matches
        #compute average color before white balance
        avg_color = np.zeros(3, dtype=np.float32)

        for img_path in unbalanced_image_paths:
            img = cv.imread(img_path)
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            for c in range(3):
                avg_color[c] += np.mean(img_rgb[:,:,c])
        avg_color /= num_images
        print(f"{os.path.basename(folder)} number of images: {num_images} ground-truth: {good_matches} precision: {precision:.3f} averagecolor: {avg_color.astype(np.uint8).tolist()}")
    total_precision = 1 - (abs(total_images - total_groundtruth) / (total_groundtruth + 1e-10))
    print(f"TOTAL number of images: {total_images} ground-truth: {total_groundtruth} precision: {total_precision:.3f}")


def cleanup_output_directory(output_dir):
    #obtain the desired output structure by removing unneeded files and folders
    #from the output directory, remove the histogram folder and its contents
    hist_dir = os.path.join(output_dir, "histograms")
    if os.path.exists(hist_dir):
        files = glob.glob(os.path.join(hist_dir, "*"))
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
        os.rmdir(hist_dir)
    #for each similar folder, remove the common_histogram.json file and substitute the original image with the white-balanced one
    similarity_folders = sorted(glob.glob(os.path.join(output_dir, "similar-*")))
    for folder in similarity_folders:
        common_hist_path = os.path.join(folder, "common_histogram.json")
        if os.path.exists(common_hist_path):
            os.remove(common_hist_path)
        #replace original images with white-balanced ones and remove white-balanced images
        wb_image_paths = sorted(glob.glob(os.path.join(folder, "wb_*.jpg")))
        for wb_img_path in wb_image_paths:
            original_img_name = os.path.basename(wb_img_path)[3:]  #remove 'wb_' prefix
            original_img_path = os.path.join(folder, original_img_name)
            if os.path.exists(original_img_path):
                os.remove(original_img_path)
            #rename white-balanced image to original name
            os.rename(wb_img_path, original_img_path)

if __name__ == "__main__":

    CHI_SQUARE_THRESHOLD = 0.545

    # #############################################################################
    # # COMMENT THIS BLOCK BEFORE SUBMISSION
    # # This is just do adjust some environment issues when running locally with vs code.
    # # If pathing problems arise, please check this block first.
    # #############################################################################
    # #viscomp2025/starter_code/75169_75214/tp1.py
    # #conda is acting up and doesn't activate in project subfolders, but only in upper directory, probably this issue
    # #is caused by vscode terminal settings, so we force the working directory to be the starter_code folder
    # #
    # #cwd must be viscomp2025/starter_code in order to create input and output directories correctly in local environment.
    # #vscode sets the cwd to the folder where the script is located, so we need to change it back to starter_code (parent folder)
    # if not os.getcwd().endswith("starter_code"):
    #     print("Changing working directory to starter_code")
    #     os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    #     print("Current working directory:", os.getcwd())
    # #set input folder (already existing) as readonly
    # os.chmod("./input", 0o555)  #read and execute permissions
    # # if the output folder isn't empty, clear it
    # if os.path.exists("./output"):
    #     files = glob.glob(os.path.join("./output", "*"))
    #     for f in files:
    #         if os.path.isfile(f):
    #             os.remove(f)
    #         elif os.path.isdir(f):
    #             import shutil
    #             shutil.rmtree(f)
    # else:
    #     os.makedirs("./output", exist_ok=True)
    # # since now we're on the folder above the one containing tp1.py, we need to adjust the input and output folder paths
    # INPUT_FOLDER_PATH = "./input"
    # OUTPUT_FOLDER_PATH = "./output"
    #
    #############################################################################
    resize_images(INPUT_FOLDER_PATH, OUTPUT_FOLDER_PATH, size=512)
    create_histograms(OUTPUT_FOLDER_PATH)
    group_by_similarity(OUTPUT_FOLDER_PATH, CHI_SQUARE_THRESHOLD)
    common_histogram_and_white_balance(OUTPUT_FOLDER_PATH)

    myMOPS_instance = myMOPS()
    img1 = cv.imread(os.path.join(INPUT_FOLDER_PATH, "109900.jpg"))
    img2 = cv.imread(os.path.join(INPUT_FOLDER_PATH, "109901.jpg"))
    mops_compared_img = match_images_MOPS(img1, img2)
    sift_compared_img = match_images_SIFT(img1, img2)
    #stack images vertically
    combined_img = np.vstack((mops_compared_img, sift_compared_img))
    cv.imwrite(os.path.join(OUTPUT_FOLDER_PATH, "my_match.jpg"), combined_img)

    check_identical_images(OUTPUT_FOLDER_PATH)
    print_statistics(OUTPUT_FOLDER_PATH)
    cleanup_output_directory(OUTPUT_FOLDER_PATH)
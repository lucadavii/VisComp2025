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
        ## TODO: substitute original image with white-balanced image
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
if __name__ == "__main__":

    CHI_SQUARE_THRESHOLD = 0.5
    #############################################################################
    # REMOVE THIS BLOCK BEFORE SUBMISSION
    #
    #############################################################################
    #viscomp2025/starter_code/75169_75214/tp1.py
    #cwd must be viscomp2025/starter_code in order to create input and output directories correctly.
    #vscode sets the cwd to the folder where the script is located, so we need to change it back to starter_code (parent folder)
    if not os.getcwd().endswith("starter_code"):
        print("Changing working directory to starter_code")
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        print("Current working directory:", os.getcwd())
    #set input folder (already existing) as readonly
    os.chmod("./input", 0o555)  #read and execute permissions
    # if the output folder isn't empty, clear it
    if os.path.exists("./output"):
        files = glob.glob(os.path.join("./output", "*"))
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                import shutil
                shutil.rmtree(f)
    else:
        os.makedirs("./output", exist_ok=True)
    #exit()
    #############################################################################
    resize_images("./input", "./output", size=512)
    create_histograms("./output")
    group_by_similarity("./output", CHI_SQUARE_THRESHOLD)
    common_histogram_and_white_balance("./output")

    myMOPS_instance = myMOPS()
    img1 = cv.imread("./input/109900.jpg")
    img2 = cv.imread("./input/109901.jpg")

    points1 = myMOPS_instance.my_track_points(cv.cvtColor(img1, cv.COLOR_BGR2GRAY), maxCorners=100, qualityLevel=0.01, minDistance=10)
    print(f"Tracked {len(points1)} points in Image 1")
    print(points1)
    
    points2 = myMOPS_instance.my_track_points(cv.cvtColor(img2, cv.COLOR_BGR2GRAY), maxCorners=100, qualityLevel=0.01, minDistance=10)
    print(f"Tracked {len(points2)} points in Image 2")
    print(points2)

    
    matched_img=myMOPS_instance.my_draw_matches(img1, img2)
    cv.imshow("My MOPS Matches", matched_img)
    cv.waitKey(0)
    cv.destroyAllWindows()



    # # Example of computing Chi-Square distance between two histograms
    # start_val, end_val = 0, 255  # Exclude first and last bins as they contain possibly over/under-exposed pixels
    # with open("./output/histograms/100000_histogram.json", "r") as f:
    #     histA = json.load(f)
    #     histA = np.array([histA["red"][start_val:end_val], histA["green"][start_val:end_val], histA["blue"][start_val:end_val]],dtype=np.float32) #First index indicates the channel
    #     #print(histA.shape)
    # with open("./output/histograms/100001_histogram.json", "r") as f:
    #     histB = json.load(f)
    #     histB = np.array([histB["red"][start_val:end_val], histB["green"][start_val:end_val], histB["blue"][start_val:end_val]],dtype=np.float32)  # Combine channels for distance calculation
    # distance = histogram_distance(histA, histB)
    # b_dist=bhattacharyya_distance(histA, histB)

    # #as a test, calculate distance between first histogram and all histograms in the folder
    # image_paths = sorted(glob.glob(os.path.join("./output/histograms/", "*_histogram.json")))
    # #print(image_paths)
    # #np array to store distances
    # distances = np.zeros((len(image_paths), 2)) #2 distances for each image
    # for i, img_path in enumerate(image_paths):
    #     with open(img_path, "r") as f:
    #         hist = json.load(f)
    #         hist = np.array([hist["red"][start_val:end_val], hist["green"][start_val:end_val], hist["blue"][start_val:end_val]],dtype=np.float32) #First index indicates the channel
    #         distances[i, 0] = histogram_distance(histA, hist)
    #         distances[i, 1] = bhattacharyya_distance(histA, hist)
    # #save distances as csv
    # np.savetxt("distances.csv", distances, delimiter=",", header="Chi-Square,Bhattacharyya", comments='')

    # target_dir = os.path.join("./output", "similar-0")
    # os.makedirs(target_dir, exist_ok=True)
    # for fname in os.listdir("./output"):
    #     src_path = os.path.join("./output", fname)
    #     if os.path.isfile(src_path):
    #         dst_path = os.path.join(target_dir, fname)
    #         # use os.replace to move/overwrite atomically and avoid issues with existing files
    #         os.replace(src_path, dst_path)

    # #plot distances in two different plots in the same figure
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.title("Chi-Square Distances from Image 100000")
    # plt.xlabel("Image Index")
    # plt.ylabel("Chi-Square Distance")
    # plt.plot(distances[:, 0], marker='o')
    # plt.subplot(2, 1, 2)
    # plt.title("Bhattacharyya Distances from Image 100000")
    # plt.xlabel("Image Index")
    # plt.ylabel("Bhattacharyya Distance")
    # plt.plot(distances[:, 1], marker='o', color='orange')
    # plt.tight_layout()
    # plt.show()

    # plot_histogram(histA, "Histogram of Image 100000")
    # plot_histogram(histB, "Histogram of Image 100001")
    # # print(f"Chi-Square distance between histograms: {distance}")
    # # print(f"Bhattacharyya distance between histograms: {b_dist}")
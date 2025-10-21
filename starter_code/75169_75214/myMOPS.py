
# #############################################
# • my_track_points : Function that find points using cv2.goodFeaturesToTrack .
# • my_point_rotation : Function that given a point and an area around the point, finds its
# rotation (use `cv2.Sobel,θ= arctan(Iy, Ix) and histogram to find dominant angle)
# • my_descriptor : Function that creates your own descriptor for each point.
# ◦ The goal is to take a window around the point (e.g.,40x40), rotate the image and
# downsample it to 8x8. The result is a flatten array of 64 float values normalized
# between 0 and 1. Explore functions such as cv2.getRotationMatrix2D and
# cv2.warpAffine and try to create the best descriptor you can. Present and explain
# the algorithm and images of the 8x8 patch in the report.
# • my_distance : Create a function that given two descriptors gives a match distance score
# using the Euclidean Distance.
# • my_match : Create a function that gives a match based on the Nearest Neighbours ratio
# and a certain Threshold.
# • output an image comparison ("my_match.jpg") in the output directory using your
# algorithm and using SIFT (e.g., Use these two "109900.jpg" "109901.jpg")
# #############################################
import cv2 as cv
import numpy as np


class myMOPS:
    def __init__(self):
        pass
    def _to_gray(self, img) -> cv.typing.MatLike:
        """ Convert image to grayscale if it is not already."""
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img.astype(np.float32)
    def my_track_points(self, img, maxCorners, qualityLevel, minDistance) -> cv.typing.MatLike:
        """ Function that find points using cv2.goodFeaturesToTrack . """
        img = self._to_gray(img)

        points = cv.goodFeaturesToTrack(img, mask=None, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance)

        return points.reshape(-1, 2)
    def my_point_rotation(self, img, point, window_size) -> float:
        """ Function that given a point and an area around the point finds its rotation"""
        # extract window using cv.getRectSubPix, which interpolates border pixels
        img = self._to_gray(img)
        x,y = float(point[0]), float(point[1])
        img_window = cv.getRectSubPix(img, (window_size, window_size), (x, y))
        # Compute gradients
        # pre blur to reduce noise
        img_window = cv.GaussianBlur(img_window, (5,5), sigmaX=1.0)
        sobelx = cv.Sobel(img_window, ddepth=cv.CV_32F, dx=1, dy=0, ksize=3)
        sobely = cv.Sobel(img_window, ddepth=cv.CV_32F, dx=0, dy=1, ksize=3)

        #compute angle histogram
        angles = np.arctan2(sobely, sobelx)
        magnitudes = np.sqrt(sobelx**2 + sobely**2)
        hist, bin_edges = np.histogram(angles.ravel(), bins=36, range=(-np.pi, np.pi),weights=magnitudes.ravel())
        
        best_idx = np.argmax(hist)
        dominant_angle = 0.5*(bin_edges[best_idx] + bin_edges[best_idx+1]) #center of the bin 
        return dominant_angle
    def my_descriptor(self, img, point, dominant_angle, window_size=40) -> np.ndarray:
        """ Function that creates your own descriptor for each point."""
        img = self._to_gray(img)
        x,y = float(point[0]), float(point[1])
        img_window = cv.getRectSubPix(img, (window_size, window_size), (x, y))

        # Rotate the window to align with dominant angle
        center = (window_size / 2, window_size / 2)
        rotation_matrix = cv.getRotationMatrix2D(center, np.degrees(-dominant_angle), scale=1.0/5.0)
        rotated_window = cv.warpAffine(img_window, rotation_matrix, (window_size, window_size), flags=cv.INTER_LINEAR)

        # Downsample to 8x8
        small_window = cv.resize(rotated_window, (8, 8), interpolation=cv.INTER_AREA)

        # Normalize between 0 and 1
        descriptor = small_window.flatten()
        descriptor = (descriptor - np.min(descriptor)) / (np.max(descriptor) - np.min(descriptor) + 1e-10)  # add small value to avoid division by zero
        return descriptor
    def my_distance(self, desc1, desc2) -> float:
        """ Create a function that given two descriptors gives a match distance score using the Euclidean Distance."""
        return np.linalg.norm(desc1 - desc2)
    def my_match(self, D1, D2, ratio_threshold) -> bool:
        """ Create a function that gives a match based on the Nearest Neighbours ratio and a certain Threshold.
            D1: list of descriptors from image 1
            D2: list of descriptors from image 2
            ratio_threshold: threshold for the ratio test
            Returns: list of matches
        """

        matches = []
        for i, desc1 in enumerate(D1):
            # Find the two closest descriptors in D2
            distances = [self.my_distance(desc1, desc2) for desc2 in D2]
            best_match = np.argmin(distances)
            best_match_distance = distances[best_match]
            dist_copy= distances.copy()
            dist_copy[best_match]=np.inf #remove best match by setting to small number
            
            second_best = np.argmin(dist_copy)
            second_best_distance = distances[second_best]
            if best_match_distance < ratio_threshold * second_best_distance:
                matches.append(cv.DMatch(_queryIdx=i, _trainIdx=best_match, _distance=best_match_distance))
        return matches

    def _points_to_keypoints(self, points) -> list:
        """ Convert points to cv2.KeyPoint objects."""
        keypoints = [cv.KeyPoint(x=float(p[0]), y=float(p[1]), size=3) for p in points]
        return keypoints
    def my_draw_matches(self,img1, img2) -> np.ndarray:
        """ Output an image comparison ("my_match.jpg") in the output directory using your algorithm and using SIFT."""
        # Find points
        points1 = self.my_track_points(img1, maxCorners=500, qualityLevel=0.01, minDistance=10)
        points2 = self.my_track_points(img2, maxCorners=500, qualityLevel=0.01, minDistance=10)

        # Compute descriptors
        descriptors1 = []
        for p in points1:
            angle = self.my_point_rotation(img1, p, window_size=40)
            desc = self.my_descriptor(img1, p, angle)
            descriptors1.append(desc)
        descriptors2 = []
        for p in points2:
            angle = self.my_point_rotation(img2, p, window_size=40)
            desc = self.my_descriptor(img2, p, angle)
            descriptors2.append(desc)

        # Match descriptors
        matches = self.my_match(descriptors1.copy(), descriptors2.copy(), ratio_threshold=0.7)
        # Convert points to keypoints
        keypoints1 = self._points_to_keypoints(points1)
        keypoints2 = self._points_to_keypoints(points2)

        # Draw matches
        print(f"Number of matches found: {len(matches)}")
        matched_image = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
        return matched_image
    
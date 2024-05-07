import cv2
import numpy as np
from Image import Image
from Harris import Harris
from Sift import Sift


class Matcher(Image):
    def __init__(self):
        super().__init__()
        self.harris = Harris()
        self.sift = Sift()

    def harris_corner_detection(self, image):
        return self.harris.harris_corner_detection_main(
            image.img_original, image.img_copy)

    def sift_feature_descriptors(self, image):
        _, descriptors, keypoints_in_cv_format = self.sift.detect_and_describe_features(
            image.img_copy)
        return keypoints_in_cv_format, descriptors

    def match_features(self, img1_descriptors, img2_descriptors, ssd_threshold=50000, ncc_threshold=0.8):
        # SSD matching with thresholding
        ssd_matches = []
        for descriptor1 in img1_descriptors:
            ssd_distances = []
            for descriptor2 in img2_descriptors:
                ssd_distance = np.sum((descriptor1 - descriptor2)**2)
                ssd_distances.append(ssd_distance)
            min_distance = np.min(ssd_distances)
            best_match_index = np.argmin(ssd_distances)
            # Only add matches that are below the SSD threshold
            if min_distance < ssd_threshold:
                ssd_matches.append(best_match_index)
            else:
                ssd_matches.append(None)  # Represent no match due to threshold

        # Normalized cross-correlation matching with thresholding
        ncc_matches = []
        for descriptor1 in img1_descriptors:
            ncc_correlations = []
            for descriptor2 in img2_descriptors:
                ncc_correlation = np.sum(
                    descriptor1 * descriptor2) / (np.linalg.norm(descriptor1) * np.linalg.norm(descriptor2))
                ncc_correlations.append(ncc_correlation)
            max_correlation = np.max(ncc_correlations)
            best_match_index = np.argmax(ncc_correlations)
            # Only add matches that are above the NCC threshold
            if max_correlation > ncc_threshold:
                ncc_matches.append(best_match_index)
            else:
                ncc_matches.append(None)  # Represent no match due to threshold

        return ssd_matches, ncc_matches

    def draw_matches(self, img1, kp1, img2, kp2, matches, img1_descriptors, img2_descriptors, max_distance=200.0):
        filtered_matches = []
        for i, match in enumerate(matches):
            if match is not None:
                dist = np.linalg.norm(
                    img1_descriptors[i] - img2_descriptors[match])
                if dist < max_distance:
                    filtered_matches.append(cv2.DMatch(i, match, dist))
        output_image = cv2.drawMatches(
            img1, kp1, img2, kp2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return output_image

    def plot_features(self, image1, image2):
        img1 = cv2.imread(image1.file_path)
        img2 = cv2.imread(image2.file_path)
        # Harris Corners
        harris_img1 = self.harris_corner_detection(image1)
        harris_img2 = self.harris_corner_detection(image2)

        # SIFT Features
        keypoints_img1, descriptors_img1 = self.sift_feature_descriptors(
            image1)
        keypoints_img2, descriptors_img2 = self.sift_feature_descriptors(
            image2)

        # Match features
        ssd_matches, ncc_matches = self.match_features(
            descriptors_img1, descriptors_img2)

        # Drawing matches
        img1_matched_ssd = self.draw_matches(
            img1, keypoints_img1, img2, keypoints_img2, ssd_matches, descriptors_img1, descriptors_img2)
        img1_matched_ncc = self.draw_matches(
            img1, keypoints_img1, img2, keypoints_img2, ncc_matches, descriptors_img1, descriptors_img2)

        self.figure.clf()  # clear the whole figure
        axs = self.figure.subplots(2, 2)

        # Plotting
        axs[0, 0].imshow(harris_img1)
        axs[0, 0].set_title('Harris Corners Image 1')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(harris_img2)
        axs[0, 1].set_title('Harris Corners Image 2')
        axs[0, 1].axis('off')

        axs[1, 0].imshow(cv2.cvtColor(img1_matched_ssd, cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title('SSD Matches')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(cv2.cvtColor(img1_matched_ncc, cv2.COLOR_BGR2RGB))
        axs[1, 1].set_title('NCC Matches')
        axs[1, 1].axis('off')

        self.draw()

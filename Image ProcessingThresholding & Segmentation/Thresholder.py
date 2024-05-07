import numpy as np
import cv2
from Image import Image


class Thresholder(Image):
    def __init__(self):
        super().__init__()
        self.methods = {"Otsu's": {
            "Local Thresholding": self.otsu_local,
            "Global Thresolding": self.otsu_global},
            "Optimal": {"Local Thresholding": self.optimal_local,
                        "Global Thresolding": self.optimal_global},
            "Spectral": {"Local Thresholding": self.spectral_local,
                         "Global Thresolding": self.spectral_global}}

    def otsu_global(self, image):
        # Compute histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()

        # Initialization
        best_thresh = 0
        best_sigma_b_squared = 0

        # Calculate cumulative sum and cumulative sum of squares
        cum_sum = np.cumsum(hist_norm)
        cum_sum_sq = np.cumsum(hist_norm * np.arange(256))

        for t in range(1, 256):
            # Class probabilities
            w0 = cum_sum[t]
            w1 = 1 - w0

            # Class means
            if w0 == 0 or w1 == 0:
                continue
            mean0 = cum_sum_sq[t] / w0
            mean1 = (cum_sum_sq[-1] - cum_sum_sq[t]) / w1

            # Between-class variance
            sigma_b_squared = w0 * w1 * ((mean0 - mean1) ** 2)

            # Update optimal threshold if variance is higher
            if sigma_b_squared > best_sigma_b_squared:
                best_sigma_b_squared = sigma_b_squared
                best_thresh = t

        # Apply threshold
        _, binary_image = cv2.threshold(
            image, best_thresh, 255, cv2.THRESH_BINARY)

        return [binary_image, best_thresh]

    def otsu_local(self, image, block_size=70):
        result = image.copy()

        # Divide the image into blocks
        for i in range(0, image.shape[0], block_size):
            for j in range(0, image.shape[1], block_size):
                block_height = min(block_size, image.shape[0] - i)
                block_width = min(block_size, image.shape[1] - j)
                sub_image = image[i:i+block_height, j:j+block_width]

                # Apply Otsu's thresholding to the block
                _, local_threshold = self.otsu_global(sub_image)
                _, thresholded_block = cv2.threshold(
                    sub_image, local_threshold, 255, cv2.THRESH_BINARY)

                # Copy the thresholded block back to the result image
                result[i:i+block_height, j:j+block_width] = thresholded_block

        return [result]

    def optimal_global(self, gray_image):
        top_left = gray_image[0][0]
        top_right = gray_image[0][-1]
        bottom_left = gray_image[-1][0]
        bottom_right = gray_image[-1][-1]

        background_pixels = [top_left, top_right, bottom_left, bottom_right]
        background_pixels = np.array(background_pixels)

        # Remove background pixels from the image
        object_pixels = gray_image[~np.isin(gray_image, background_pixels)]

        # Initialize threshold values
        T_old = 0
        T_new = np.percentile(gray_image, 50)
        convergence_threshold = 0.01

        # Iterate until convergence
        while abs(T_old - T_new) > convergence_threshold:
            T_old = T_new

            # Partition the image based on the threshold
            object_pixels = gray_image[gray_image > T_new]
            background_pixels = gray_image[gray_image <= T_new]

            # Calculate mean values for object and background
            mean_object = np.mean(object_pixels) if len(
                object_pixels) > 0 else 0
            mean_background = np.mean(background_pixels) if len(
                background_pixels) > 0 else 0

            # Update the threshold value
            T_new = (mean_object + mean_background) // 2

        # Create binary image using the final threshold value
        binary_image = np.zeros_like(gray_image)
        binary_image[gray_image > T_new] = 255

        return [binary_image, T_new]

    def optimal_local(self, gray_image, block_size=70):
        result = gray_image.copy()

        # Divide the gray_image into blocks
        for i in range(0, gray_image.shape[0], block_size):
            for j in range(0, gray_image.shape[1], block_size):
                block_height = min(block_size, gray_image.shape[0] - i)
                block_width = min(block_size, gray_image.shape[1] - j)
                sub_image = gray_image[i:i+block_height, j:j+block_width]

                # Apply Otsu's thresholding to the block
                _, local_threshold = self.optimal_global(
                    sub_image)
                _, thresholded_block = cv2.threshold(
                    sub_image, local_threshold, 255, cv2.THRESH_BINARY)

                # Copy the thresholded block back to the result image
                result[i:i+block_height, j:j+block_width] = thresholded_block

        return [result]

    def spectral_global(self, img):
        # Apply spectral thresholding
        thresholds = self.spectral_thresholding(img)

        # Apply double thresholding
        thresholded_img = self.double_spectral_thresholding(img, thresholds)

        return [thresholded_img]

    def spectral_local(self, image, block_size=70):
        local_threshold = np.zeros_like(image)
        for i in range(0, image.shape[0], block_size):
            for j in range(0, image.shape[1], block_size):
                block_height = min(block_size, image.shape[0] - i)
                block_width = min(block_size, image.shape[1] - j)
                sub_image = image[i:i+block_height, j:j+block_width]

                thresholds = self.spectral_thresholding(sub_image)

                local_threshold[i:i+block_height, j:j +
                                block_width] = self.double_spectral_thresholding(sub_image, thresholds)

        return [local_threshold]

    def spectral_thresholding(self, image):
        # Calculate the histogram of the input image
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

        # Calculate the cumulative sum of the histogram values
        cumulative_sum = np.cumsum(histogram)

        # Calculate the total number of pixels in the image
        total = image.shape[0] * image.shape[1]

        # Set the threshold values
        tau1 = 0.0
        tau2 = 0.0

        # Set to 0.5 to balance the trade-off between the two threshold values
        alpha = 0.5

        background_pixels = 0
        foreground_pixels = 0
        background_pixels_sum = 0.0
        foreground_pixels_sum = 0.0
        max_variance = 0.0
        threshold_1 = 0
        threshold_2 = 0

        # Iterates over all possible threshold values from 0 to 255
        for i in range(256):
            background_pixels += histogram[i]

            if background_pixels == 0:
                continue

            foreground_pixels = total - background_pixels

            if foreground_pixels == 0:
                break

            background_pixels_sum += i * histogram[i]
            foreground_pixels_sum = cumulative_sum[-1] - background_pixels_sum

            background_mean = background_pixels_sum / background_pixels
            foreground_mean = foreground_pixels_sum / foreground_pixels

            # Calculates the within-class variance and between-class variance
            # Measures the separation between classes
            variance_difference = background_pixels * foreground_pixels * \
                (background_mean - foreground_mean) ** 2

            # Updates the threshold values if the between-class variance is the max variance
            if variance_difference > max_variance:
                max_variance = variance_difference
                threshold_1 = i

            # This line accumulates the sum of pixel intensities up to intensity level i. It will be used to determine when to update the value of tau2.
            tau1 += histogram[i] * i

            # Updates tau2
            if tau1 > alpha * total and tau2 == 0:
                tau2 = i

        # Threshold_2 is the average of threshold_1 and tau2
        threshold_2 = round((threshold_1 + tau2) / 2.0)

        # Compensate the threshold values
        threshold_1 -= 50
        threshold_2 -= 50

        return threshold_1, threshold_2

    def double_spectral_thresholding(self, img, thresholds):
        # Apply the threshold to the image
        thresholded_img = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] <= thresholds[1]:
                    thresholded_img[i, j] = 0
                elif thresholds[1] < img[i, j] <= thresholds[0]:
                    thresholded_img[i, j] = 128
                else:
                    thresholded_img[i, j] = 255
        return thresholded_img

    def apply_thresholding(self, image, thresholding_method, thresholding_type):
        result = self.methods[thresholding_method][thresholding_type](image)
        return result[0]

from math import floor
from numpy import all, sqrt, floor, round
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Image import Image
from cv2 import subtract


class Sift(Image):
    def __init__(self):
        super().__init__()

    def create_scale_space(self, image, num_octaves, num_scales, k=sqrt(2)):
        scale_space_images = []
        for octave in range(num_octaves):
            octave_images = []
            sigma = 1.6
            for scale in range(num_scales):
                sigma_current = sigma * (k ** scale)
                blurred_image = cv2.GaussianBlur(
                    image, (0, 0), sigmaX=sigma_current, sigmaY=sigma_current)
                octave_images.append(blurred_image)
            scale_space_images.append(octave_images)
            image = cv2.resize(image, (int(
                image.shape[1] / 2), int(image.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
        return scale_space_images

    def diff_of_gaussian(self, scale_space_images):
        diff_of_gaussian_result = []
        for octave in scale_space_images:
            octave_images = []
            for first_image, second_image in zip(octave, octave[1:]):
                octave_images.append(subtract(second_image, first_image))
            diff_of_gaussian_result.append(octave_images)
        return diff_of_gaussian_result

    def keypoint_localization(self, dog_res, contrast_threshold=0.04, num_intervals=3):
        keypoints = []
        threshold = floor(0.5 * contrast_threshold / num_intervals * 255)
        for octave_idx, octave in enumerate(dog_res):
            for scale_idx in range(1, len(octave) - 1):
                for i in range(1, octave[scale_idx].shape[0] - 1):
                    for j in range(1, octave[scale_idx].shape[1] - 1):
                        pixel_value = octave[scale_idx][i, j]
                        neighbors = [octave[scale_idx - 1][i - 1:i + 2, j - 1:j + 2],
                                     octave[scale_idx][i -
                                                       1:i + 2, j - 1:j + 2],
                                     octave[scale_idx + 1][i - 1:i + 2, j - 1:j + 2]]
                        if self.is_local_extremum(neighbors, threshold):
                            keypoints.append(
                                (i, j, octave[scale_idx], octave_idx, scale_idx, pixel_value))
        return keypoints

    def is_local_extremum(self, neighbors, threshold):
        first_subimage, second_subimage, third_subimage = neighbors
        center_pixel_value = second_subimage[1, 1]
        if abs(center_pixel_value) > threshold:
            if center_pixel_value > 0:
                return all(center_pixel_value >= first_subimage) and \
                    all(center_pixel_value >= third_subimage) and \
                    all(center_pixel_value >= second_subimage[0, :]) and \
                    all(center_pixel_value >= second_subimage[2, :]) and \
                    center_pixel_value >= second_subimage[1, 0] and \
                    center_pixel_value >= second_subimage[1, 2]
            elif center_pixel_value < 0:
                return all(center_pixel_value <= first_subimage) and \
                    all(center_pixel_value <= third_subimage) and \
                    all(center_pixel_value <= second_subimage[0, :]) and \
                    all(center_pixel_value <= second_subimage[2, :]) and \
                    center_pixel_value <= second_subimage[1, 0] and \
                    center_pixel_value <= second_subimage[1, 2]
        return False

    def keypoint_selection(self, keypoints, gaussian_images, contrast_threshold=0.03, eigenvalue_ratio=10, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
        selected_keypoints_with_orientations = []
        for keypoint in keypoints:
            i, j, image, octave_idx, scale_idx, pixel_value = keypoint
            dx = np.gradient(image, axis=1)
            dy = np.gradient(image, axis=0)
            dxx = np.gradient(dx, axis=1)
            dyy = np.gradient(dy, axis=0)
            dxy = np.gradient(dx, axis=0)
            xy_hessian = np.array(
                [[dxx[i, j], dxy[i, j]], [dxy[i, j], dyy[i, j]]])
            contrast = abs(pixel_value / 255)
            xy_hessian_det = np.linalg.det(xy_hessian)
            xy_hessian_trace = np.trace(xy_hessian)
            if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
                continue
            if contrast < contrast_threshold:
                continue
            keypoints_with_orientations = self.computeKeypointsWithOrientations(
                keypoint, octave_idx, gaussian_images[octave_idx][scale_idx], radius_factor, num_bins, peak_ratio, scale_factor)
            selected_keypoints_with_orientations.extend(
                keypoints_with_orientations)
        return selected_keypoints_with_orientations

    def computeKeypointsWithOrientations(self, keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=8, peak_ratio=0.8, scale_factor=1.5):
        keypoints_with_orientations = []
        image_height, image_width = gaussian_image.shape
        scale_idx = keypoint[4]
        scale = scale_factor * scale_idx / (2 ** (octave_index))
        radius = int(round(radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)
        histogram = np.zeros(num_bins)
        for i in range(-radius, radius + 1):
            region_y = int(round(keypoint[1] / (2 ** octave_index))) + i
            if 0 < region_y < image_height - 1:
                for j in range(-radius, radius + 1):
                    region_x = int(
                        round(keypoint[0] / (2 ** octave_index))) + j
                    if 0 < region_x < image_width - 1:
                        dx = gaussian_image[region_y, region_x + 1] - \
                            gaussian_image[region_y, region_x - 1]
                        dy = gaussian_image[region_y - 1, region_x] - \
                            gaussian_image[region_y + 1, region_x]
                        gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                        weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                        histogram_index = int(
                            round(gradient_orientation * num_bins / 360))
                        histogram[histogram_index %
                                  num_bins] += weight * gradient_magnitude
        smooth_histogram = np.convolve(
            histogram, np.array([1, 4, 6, 4, 1]) / 16, mode='same')
        orientation_max = np.max(smooth_histogram)
        orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(
            smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
        for peak_index in orientation_peaks:
            peak_value = smooth_histogram[peak_index]
            if peak_value >= peak_ratio * orientation_max:
                left_value = smooth_histogram[(peak_index - 1) % num_bins]
                right_value = smooth_histogram[(peak_index + 1) % num_bins]
                interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (
                    left_value - 2 * peak_value + right_value)) % num_bins
                orientation = 360 - interpolated_peak_index * 360 / num_bins
                if abs(orientation - 360) < 1e-5:
                    orientation = 0
                keypoints_with_orientations.append(
                    (keypoint[0], keypoint[1], keypoint[2], keypoint[3], keypoint[4], keypoint[5], orientation))
        return keypoints_with_orientations

    def generateDescriptors(self, keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
        descriptors = []
        for keypoint in keypoints:
            x, y, _, octave_idx, scale_idx, _, orientation = keypoint
            scale = 1.5 * scale_idx / (2 ** (octave_idx))
            gaussian_image = gaussian_images[octave_idx][scale_idx]
            num_rows, num_cols = gaussian_image.shape
            point = np.array([x, y], dtype=int)
            bins_per_degree = num_bins / 360.
            angle = 360. - orientation
            cos_angle = np.cos(np.deg2rad(angle))
            sin_angle = np.sin(np.deg2rad(angle))
            weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
            row_bin_list = []
            col_bin_list = []
            magnitude_list = []
            orientation_bin_list = []
            histogram_tensor = np.zeros(
                (window_width + 2, window_width + 2, num_bins))
            hist_width = scale_multiplier * 0.5 * scale
            half_width = int(round(hist_width * np.sqrt(2)
                                   * (window_width + 1) * 0.5))
            half_width = int(
                min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))
            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * sin_angle + row * cos_angle
                    col_rot = col * cos_angle - row * sin_angle
                    row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                    col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                    if 0 <= row_bin < window_width and 0 <= col_bin < window_width:
                        window_row = int(round(point[1] + row))
                        window_col = int(round(point[0] + col))
                        if 0 < window_row < num_rows - 1 and 0 < window_col < num_cols - 1:
                            dx = gaussian_image[window_row, window_col + 1] - \
                                gaussian_image[window_row, window_col - 1]
                            dy = gaussian_image[window_row - 1, window_col] - \
                                gaussian_image[window_row + 1, window_col]
                            gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
                            gradient_orientation = np.rad2deg(
                                np.arctan2(dy, dx)) % 360
                            weight = np.exp(
                                weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            magnitude_list.append(weight * gradient_magnitude)
                            orientation_bin_list.append(
                                (gradient_orientation - angle) * bins_per_degree)
            for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
                row_bin_floor, col_bin_floor, orientation_bin_floor = int(
                    floor(row_bin)), int(floor(col_bin)), int(floor(orientation_bin))
                row_fraction, col_fraction, orientation_fraction = row_bin - \
                    row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
                orientation_bin_floor %= num_bins
                c1 = magnitude * row_fraction
                c0 = magnitude * (1 - row_fraction)
                c11 = c1 * col_fraction
                c10 = c1 * (1 - col_fraction)
                c01 = c0 * col_fraction
                c00 = c0 * (1 - col_fraction)
                c111 = c11 * orientation_fraction
                c110 = c11 * (1 - orientation_fraction)
                c101 = c10 * orientation_fraction
                c100 = c10 * (1 - orientation_fraction)
                c011 = c01 * orientation_fraction
                c010 = c01 * (1 - orientation_fraction)
                c001 = c00 * orientation_fraction
                c000 = c00 * (1 - orientation_fraction)
                histogram_tensor[row_bin_floor + 1,
                                 col_bin_floor + 1, orientation_bin_floor] += c000
                histogram_tensor[row_bin_floor + 1, col_bin_floor +
                                 1, (orientation_bin_floor + 1) % num_bins] += c001
                histogram_tensor[row_bin_floor + 1,
                                 col_bin_floor + 2, orientation_bin_floor] += c010
                histogram_tensor[row_bin_floor + 1, col_bin_floor +
                                 2, (orientation_bin_floor + 1) % num_bins] += c011
                histogram_tensor[row_bin_floor + 2,
                                 col_bin_floor + 1, orientation_bin_floor] += c100
                histogram_tensor[row_bin_floor + 2, col_bin_floor +
                                 1, (orientation_bin_floor + 1) % num_bins] += c101
                histogram_tensor[row_bin_floor + 2,
                                 col_bin_floor + 2, orientation_bin_floor] += c110
                histogram_tensor[row_bin_floor + 2, col_bin_floor +
                                 2, (orientation_bin_floor + 1) % num_bins] += c111
            descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()
            threshold = norm(descriptor_vector) * descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(norm(descriptor_vector), 1e-6)
            descriptor_vector = np.round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)
        return np.array(descriptors, dtype='float32')

    def display_scale_space(self, scale_space):
        num_octaves, num_scales = len(scale_space), len(scale_space[0])
        scale_space_images = []
        for i in range(num_octaves):
            octave_images = []
            for j in range(num_scales):
                image = scale_space[i][j]
                max_dim = max(image.shape)
                tick_spacing = max_dim // 4
                fig, ax = plt.subplots(figsize=(max_dim / 100, max_dim / 100))
                ax.imshow(image, cmap='gray')
                ax.set_xticks(np.arange(0, image.shape[1], tick_spacing))
                ax.set_yticks(np.arange(0, image.shape[0], tick_spacing))
                ax.set_title(f'Octave {i + 1}, Scale {j + 1}')
                ax.axis('off')
                fig.canvas.draw()
                image_data = np.frombuffer(
                    fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_data = image_data.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,))
                octave_images.append(image_data)
                plt.close(fig)
            scale_space_images.append(octave_images)
        return scale_space_images

    def display_diff_of_gaussian(self, diff_of_gaussian_result):
        num_octaves, num_scales = len(diff_of_gaussian_result), len(
            diff_of_gaussian_result[0])
        dog_images = []
        for i in range(num_octaves):
            octave_images = []
            for j in range(num_scales):
                image = diff_of_gaussian_result[i][j]
                max_dim = max(image.shape)
                tick_spacing = max_dim // 4
                fig, ax = plt.subplots(figsize=(max_dim / 100, max_dim / 100))
                ax.imshow(image, cmap='gray')
                ax.set_xticks(np.arange(0, image.shape[1], tick_spacing))
                ax.set_yticks(np.arange(0, image.shape[0], tick_spacing))
                ax.set_title(f'Octave {i + 1}, Scale {j + 1}')
                ax.axis('off')
                fig.canvas.draw()
                image_data = np.frombuffer(
                    fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_data = image_data.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,))
                octave_images.append(image_data)
                plt.close(fig)
            dog_images.append(octave_images)
        return dog_images

    def display_keypoints(self, image, keypoints):
        height, width = image.shape
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.imshow(image, cmap='gray')
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        for keypoint in keypoints:
            x, y, _, _, _, _ = keypoint
            ax.scatter(x, y, color='red', s=10, marker='x')
        ax.axis('off')
        fig.canvas.draw()
        image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_data = image_data.reshape(
            fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image_data

    def display_keypoints_with_orientations(self, image, keypoints):
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(
            255, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return image_with_keypoints

    def generateKeyPoints(self, keypoints_info, size_multiplier=10):
        keypoints = []
        for info in keypoints_info:
            i, j, _, octave_idx, scale_idx, _, orientation = info
            scale = size_multiplier * (1.5 * scale_idx / (2 ** octave_idx))
            pt = (j, i)
            # Create a cv2.KeyPoint object for each keypoint
            kp = cv2.KeyPoint(x=pt[0], y=pt[1],
                              size=scale, angle=orientation)
            keypoints.append(kp)
        return keypoints

    def detect_and_describe_features(self, image, num_octaves=4, num_scales=5, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
        image_float = image.astype('float32')
        scale_space = self.create_scale_space(
            image_float, num_octaves, num_scales)

        Dog = self.diff_of_gaussian(scale_space)

        keypoint_detection = self.keypoint_localization(Dog)

        selected_keypoints_with_orientations = self.keypoint_selection(
            keypoint_detection, scale_space)

        descriptors = self.generateDescriptors(selected_keypoints_with_orientations,
                                               scale_space, window_width, num_bins, scale_multiplier, descriptor_max_value)

        keypoints_in_cv_format = self.generateKeyPoints(
            selected_keypoints_with_orientations)

        return selected_keypoints_with_orientations, descriptors, keypoints_in_cv_format

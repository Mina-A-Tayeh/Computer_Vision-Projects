from Image import Image
import numpy as np
import cv2
import scipy.ndimage
from scipy.ndimage import convolve


class Canny(Image):
    def __init__(self):
        super().__init__()

    def gaussian_filter(self, sigma):
        print("gaussian")
        x, y = np.meshgrid(np.linspace(-1, 1, 3),
                           np.linspace(-1, 1, 3))

        d = np.sqrt(x * x + y * y)

        mu = 0.0
        gaussian_filter_matrix = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

        gaussian_filter_matrix *= 1 / np.sum(gaussian_filter_matrix)

        image_gaussian = convolve(
            self.hough_img_input.img_copy, gaussian_filter_matrix)

        return image_gaussian

    def sobel_filters(self, smoothed_image):
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]])
        Ix = cv2.filter2D(smoothed_image, -1, kernel_x)
        Iy = cv2.filter2D(smoothed_image, -1, kernel_y)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)

        return G, theta

    def non_maximum_suppression(self, gradient_magnitude, gradient_direction):
        M, N = gradient_magnitude.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = gradient_direction * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = gradient_magnitude[i, j + 1]
                        r = gradient_magnitude[i, j - 1]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = gradient_magnitude[i + 1, j - 1]
                        r = gradient_magnitude[i - 1, j + 1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = gradient_magnitude[i + 1, j]
                        r = gradient_magnitude[i - 1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = gradient_magnitude[i - 1, j - 1]
                        r = gradient_magnitude[i + 1, j + 1]

                    if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                        Z[i, j] = gradient_magnitude[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        return Z

    def double_threshold(self, image, low_threshold_ratio=0.05, high_threshold_ratio=0.15):
        high_threshold = image.max() * high_threshold_ratio
        low_threshold = high_threshold * low_threshold_ratio

        M, N = image.shape
        res = np.zeros((M, N), dtype=np.int32)

        weak = np.int32(50)
        strong = np.int32(255)

        strong_i, strong_j = np.where(image >= high_threshold)
        zeros_i, zeros_j = np.where(image < low_threshold)

        weak_i, weak_j = np.where(
            (image <= high_threshold) & (image >= low_threshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res, weak, strong

    def hysteresis(self, image, weak, strong=255):
        M, N = image.shape
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if (image[i, j] == weak):
                    try:
                        if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (
                                image[i + 1, j + 1] == strong)
                                or (image[i, j - 1] == strong) or (image[i, j + 1] == strong)
                                or (image[i - 1, j - 1] == strong) or (image[i - 1, j] == strong) or (
                                image[i - 1, j + 1] == strong)):
                            image[i, j] = strong
                        else:
                            image[i, j] = 0
                    except IndexError as e:
                        pass
        return image

    def canny_edge_detector(self, sigma=2.0, low_threshold_ratio=0.01,
                            high_threshold_ratio=0.05):
        smoothed_image = Canny.gaussian_filter(self, sigma)
        gradient_magnitude, gradient_direction = Canny.sobel_filters(
            self, smoothed_image)
        print("sobel")
        non_max_image = Canny.non_maximum_suppression(
            self, gradient_magnitude, gradient_direction)
        threshold_image, weak, strong = Canny.double_threshold(
            self, non_max_image, low_threshold_ratio, high_threshold_ratio)
        edge_image = Canny.hysteresis(self, threshold_image, weak, strong)
        return edge_image

import cv2
import numpy as np
from Image import Image


class ImageEnhancer(Image):
    def __init__(self):
        super().__init__()

    def equalize_image(self):
        # get the histogram of the image
        hist, bins = np.histogram(self.img_copy.flatten(), 256, [0, 256])

        # get the Cumulative Distribution Function
        cdf = hist.cumsum()

        # normalization
        cdf_normalized = cdf * (255 / cdf.max())

        # mapping form original image to the cdf
        equalized_image = np.interp(
            self.img_copy.flatten(), bins[:-1], cdf_normalized).reshape(self.img_copy.shape)
        equalized_image = equalized_image.astype(np.uint8)

        return equalized_image

    def normalize_image(self):
        min_val = np.min(self.img_copy)
        max_val = np.max(self.img_copy)
        normalized_image = ((self.img_copy - min_val) /
                            (max_val - min_val)) * 255

        # Convert to uint8 data type (required for image display)
        normalized_image = normalized_image.astype(np.uint8)
        return normalized_image

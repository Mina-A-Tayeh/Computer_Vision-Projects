import numpy as np
import cv2
from Image import Image


class Thresholder(Image):
    def __init__(self):
        super().__init__()
        self.global_thresh_value = 128
        self.block_size = 11
        self.c = 2
        self.max_value = 255

    def manual_global_threshold(self):
        # Initialize the output image with zeros (all black)
        thresh_image = np.zeros_like(self.img_copy)

        # Get the dimensions of the image
        rows, cols = self.img_copy.shape

        # Loop over each pixel in the image
        for i in range(rows):
            for j in range(cols):
                # Apply the threshold
                if self.img_copy[i, j] >= self.global_thresh_value:
                    thresh_image[i, j] = self.max_value
                else:
                    thresh_image[i, j] = 0

        return thresh_image

    def manual_local_threshold(self):
        # Create an image to store the thresholded result
        thresh_image = np.zeros_like(self.img_copy)

        # Ensure block_size is odd
        if self.block_size % 2 == 0:
            self.block_size += 1

        # Pad the image to handle the border pixels
        pad_size = self.block_size // 2
        padded_image = cv2.copyMakeBorder(
            self.img_copy, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)

        # Iterate over each pixel in the image
        for i in range(pad_size, padded_image.shape[0] - pad_size):
            for j in range(pad_size, padded_image.shape[1] - pad_size):
                # Define the current block for the pixel
                block = padded_image[i-pad_size:i +
                                     pad_size+1, j-pad_size:j+pad_size+1]

                # Calculate the mean of the block
                mean = block.mean()

                # Apply the adaptive thresholding
                if self.img_copy[i - pad_size, j - pad_size] > mean - self.c:
                    thresh_image[i - pad_size, j - pad_size] = self.max_value
                else:
                    thresh_image[i - pad_size, j - pad_size] = 0

        return thresh_image

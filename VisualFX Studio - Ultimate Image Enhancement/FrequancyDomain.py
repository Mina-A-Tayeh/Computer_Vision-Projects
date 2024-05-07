import numpy as np
import math
from scipy.fft import ifft2, fftshift, ifftshift, fft2
from Image import Image


class FrequencyDomain(Image):
    def __init__(self):
        super().__init__()

    def get_max_freq(self):
        # Compute the 2D Fourier Transform
        fft = fft2(self.img_copy)

        # Shift the zero-frequency component to the center
        fft_shifted = fftshift(fft)

        # Compute the magnitude of the spectrum
        mag = np.abs(fft_shifted)

        # Find the maximum magnitude frequency
        min_freq_index = np.unravel_index(np.argmin(mag), mag.shape)
        min_freq_row, min_freq_col = min_freq_index
        # Calculate the offset from the center
        rows, cols = mag.shape
        center_row, center_col = rows // 2, cols // 2
        min_freq_row_offset = abs(min_freq_row - center_row)
        min_freq_col_offset = abs(min_freq_col - center_col)

        self.max_freq = min(min_freq_col_offset, min_freq_row_offset)

        return self.max_freq

    def low_pass_filter(self, image, cutoff_freq):
        rows, cols = image.shape
        center_row, center_col = rows // 2, cols // 2
        mask = np.zeros_like(image)
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - center_row)**2 + (j - center_col)**2) <= cutoff_freq:
                    mask[i, j] = 1
        return mask

    def high_pass_filter(self, image, cutoff_freq):
        rows, cols = image.shape
        center_row, center_col = rows // 2, cols // 2
        mask = np.ones_like(image)
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - center_row)**2 + (j - center_col)**2) <= cutoff_freq:
                    mask[i, j] = 0
        return mask

    def apply_filter(self, cutoff_freq, filter_type='low_pass'):
        # Compute the Discrete Fourier Transform of the image
        fft_shifted = fftshift(fft2(self.img_copy))

        # Apply the selected filter
        if filter_type == 'low_pass':
            mask = self.low_pass_filter(fft_shifted, cutoff_freq)
        elif filter_type == 'high_pass':
            mask = self.high_pass_filter(fft_shifted, cutoff_freq)

        # Apply the filter to the Fourier spectrum
        filtered_spectrum = fft_shifted * mask

        return filtered_spectrum

    def inverse_fourier(self, filtered_spectrum):
      # Restore the image after filtering
        filtered_spectrum_shifted = ifftshift(filtered_spectrum)
        restored_image = ifft2(filtered_spectrum_shifted).real.astype(np.uint8)

        return restored_image

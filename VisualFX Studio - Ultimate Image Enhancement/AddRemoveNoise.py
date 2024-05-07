import numpy as np
import cv2
from Image import Image


class AddRemoveNoise(Image):
    def __init__(self):
        super().__init__()
        self.kernel_rows = 11  # to be changed
        self.kernel_columns = 11  # to be changed
        self.kernel_size = 11  # to be changed
        self.selected_noise = 'None'
        self.selected_filter = 'None'
        self.add_noise_filters = {'Uniform': {'func': self.add_uniform_noise,
                                              'n_sliders': 1,
                                              'ranges': [[0, 255]],
                                              'steps': [1],
                                              'inputs': ['Noise range: ']},
                                  'Gaussian': {'func': self.add_gaussian_noise,
                                               'n_sliders': 2,
                                               'ranges': [[0, 255], [0, 50]],
                                               'steps': [1, 1],
                                               'inputs': ['Mean: ', 'Std deviation: ']},
                                  'Salt & Pepper': {'func': self.add_salt_and_pepper_noise,
                                                    'n_sliders': 2,
                                                    'ranges': [[0, 100], [0, 100]],
                                                    'steps': [1, 1],
                                                    'inputs': ['Salt: ', 'pepper: ']}}
        self.remove_noise_filters = {'Avarage': {'func': self.remove_average_noise},
                                     'Gaussian': {'func': self.remove_gaussian_noise},
                                     'Median': {'func': self.remove_median_noise}}

    def add_uniform_noise(self, noise_range):
        self.noise_image = self.img_copy.copy()
        noise = np.random.uniform(-noise_range / 2,
                                  noise_range / 2, size=self.noise_image.shape)
        self.noise_image = np.clip(
            self.noise_image + noise, 0, 255).astype(np.uint8)
        return self.noise_image

    def add_gaussian_noise(self, mean, std_dev):
        self.noise_image = self.img_copy.copy()
        noise = np.random.normal(mean, std_dev, size=self.noise_image.shape)
        self.noise_image = np.clip(
            self.noise_image + noise, 0, 255).astype(np.uint8)
        return self.noise_image

    def add_salt_and_pepper_noise(self, salt_prob, pepper_prob):
        self.noise_image = self.img_copy.copy()
        salt_thresh = 1 - salt_prob
        pepper_thresh = pepper_prob
        noise = np.random.rand(*self.noise_image.shape)
        self.noise_image[noise > salt_thresh] = 255
        self.noise_image[noise < pepper_thresh] = 0
        return self.noise_image

    def remove_average_noise(self):
        average_filter_matrix = np.ones(
            (self.kernel_rows, self.kernel_columns)) * (1/(self.kernel_columns * self.kernel_rows))

        image_average = cv2.filter2D(
            src=self.noise_image, kernel=average_filter_matrix, ddepth=-1)
        return image_average

    def remove_gaussian_noise(self):
        x, y = np.meshgrid(np.linspace(-1, 1, self.kernel_rows),
                           np.linspace(-1, 1, self.kernel_columns))
        d = np.sqrt(x*x + y*y)
        sigma, mu = 1.0, 0.0

        gaussian_filter_matrix = np.exp(-((d - mu)**2 / (2.0 * sigma**2)))

        gaussian_filter_matrix *= 1/np.sum(gaussian_filter_matrix)

        image_gaussian = cv2.filter2D(
            src=self.noise_image, kernel=gaussian_filter_matrix, ddepth=-1)
        return image_gaussian

    def remove_median_noise(self):
        noise_image = self.img_copy
        height, width = noise_image.shape
        padding_size = self.kernel_size // 2

        padded_image = np.pad(noise_image, padding_size, mode='constant')

        filtered_image = np.zeros_like(noise_image)
        for i in range(height):
            for j in range(width):
                kernel = padded_image[i:i+self.kernel_size,
                                      j:j+self.kernel_size]
                filtered_image[i][j] = np.median(kernel)

        return filtered_image

    def handle_ui(self, selected_noise, noise_sliders, noise_labels):
        self.selected_noise = selected_noise
        n_sliders = self.add_noise_filters[selected_noise]['n_sliders']
        ranges = self.add_noise_filters[selected_noise]['ranges']
        steps = self.add_noise_filters[selected_noise]['steps']
        inputs = self.add_noise_filters[selected_noise]['inputs']

        if n_sliders == 1:
            noise_sliders[1].hide()
            noise_labels[1].hide()
        else:
            noise_sliders[1].show()
            noise_labels[1].show()

        for i in range(n_sliders):
            noise_sliders[i].setMinimum(ranges[i][0])
            noise_sliders[i].setMaximum(ranges[i][1])
            noise_sliders[i].setSingleStep(steps[i])
            noise_labels[i].setText(f"{inputs[i]}0")
            noise_sliders[i].setValue(1)

    def handle_ui_filter(self, selected_filter):
        print(selected_filter)
        self.selected_filter = selected_filter
        return self.remove_noise_filters[f"{selected_filter}"]['func']()

    def handle_slider_change(self, slider_values, slider_labels):
        inputs = self.add_noise_filters[self.selected_noise]['inputs']
        if self.selected_noise == 'Salt & Pepper':
            for i in range(len(slider_values)):
                slider_values[i] /= 100
        for i in range(len(slider_values)):
            slider_labels[i-1].setText(f"{inputs[i-1]} {slider_values[i-1]}")

        if self.selected_noise == 'Uniform':
            return self.add_noise_filters[self.selected_noise]['func'](slider_values[0])
        else:
            return self.add_noise_filters[self.selected_noise]['func'](slider_values[0], slider_values[1])

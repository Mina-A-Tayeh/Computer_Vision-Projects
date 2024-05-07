import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from Image import Image


class ImageStats(Image):
    def __init__(self, mode='gray'):
        super().__init__()
        self.mode = mode

    def calculate_histogram_rgb(self):
        image = cv2.imread(self.file_path)
        color = ('b', 'g', 'r')
        hist = np.zeros((256, 3))

        for i, col in enumerate(color):
            hist[:, i] = cv2.calcHist(
                [image], [i], None, [256], [0, 256]).flatten()

        return hist

    def calculate_histogram_gray(self):
        image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        return hist

    def plot_histogram(self):
        self.figure.clear()
        if self.mode == 'rgb':
            hist = self.calculate_histogram_rgb()
            for i in range(3):
                ax = self.figure.add_subplot(3, 1, i+1)
                ax.bar(range(256), hist[:, i], color=('b', 'g', 'r')[i])
                ax.set_xlim([0, 256])
        elif self.mode == 'gray':
            hist = self.calculate_histogram_gray()
            ax = self.figure.add_subplot(1, 1, 1)
            ax.bar(range(256), hist, color='gray')
            ax.set_xlim([0, 256])

        self.draw()

    def plot_distribution_curve(self):
        self.figure.clear()
        if self.mode == 'rgb':
            image = cv2.imread(self.file_path)
            flattened_image = image.reshape((-1, 3))
            hist = np.zeros((256, 3))

            for i in range(3):
                hist[:, i] = np.histogram(flattened_image[:, i], bins=256, range=[0, 256])[0]

            ax1 = self.figure.add_subplot(1, 1, 1)
            ax1.plot(hist[:, 0], color='r', label='Red')
            ax1.plot(hist[:, 1], color='g', label='Green')
            ax1.plot(hist[:, 2], color='b', label='Blue')
            ax1.set_xlim([0, 256])

        elif self.mode == 'gray':
            image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
            hist = np.histogram(image, bins=256, range=[0, 256])[0]  # Calculate histogram for grayscale image
            ax1 = self.figure.add_subplot(1, 1, 1)
            ax1.plot(hist, color='black', label='Grayscale')  # Plot histogram as a line plot
            ax1.set_xlim([0, 256])


        self.draw()

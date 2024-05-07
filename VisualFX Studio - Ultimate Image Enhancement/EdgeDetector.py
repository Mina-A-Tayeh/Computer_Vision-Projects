import numpy as np
import cv2
from Image import Image


class EdgeDetector(Image):
    def __init__(self):
        super().__init__()
        self.edge_detectors = {'Sobel Edge Detection': self.sobel,
                               'Roberts Edge Detection': self.robert,
                               'Prewitt Edge Detection': self.prewitt,
                               'Canny Edge Detection':self.canny}

    def sobel(self):
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]])
        img_x = cv2.filter2D(self.img_copy, -1, kernel_x)
        img_y = cv2.filter2D(self.img_copy, -1, kernel_y)
        return img_x, img_y

    def robert(self):
        kernel_x = np.array([[1, 0],
                            [0, -1]])
        kernel_y = np.array([[0, 1],
                            [-1, 0]])
        img_x = cv2.filter2D(self.img_copy, -1, kernel_x)
        img_y = cv2.filter2D(self.img_copy, -1, kernel_y)
        return img_x, img_y

    def prewitt(self):
        kernel_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]])
        img_x = cv2.filter2D(self.img_copy, -1, kernel_x)
        img_y = cv2.filter2D(self.img_copy, -1, kernel_y)
        return img_x, img_y
    
    def canny(self):
        edges = cv2.Canny(self.img_copy, 100, 200, apertureSize=3)
        return edges

    def handle_edge_change(self, selected_type):
        return self.edge_detectors[selected_type]()

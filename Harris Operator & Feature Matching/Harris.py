from Image import Image
import numpy as np
import cv2


class Harris(Image):
    def __init__(self):
        super().__init__()

    def calculate_Ix_Iy(self, image):
        Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return Ix, Iy

    def calculate_Ixx_Iyy_Ixy(self, Ix, Iy):
        Ixx = Ix**2
        Iyy = Iy**2
        Ixy = Ix*Iy
        return Ixx, Iyy, Ixy

    def calculate_matrix_M(self, image, Ixx, Iyy, Ixy, window_size=3, k=0.04):
        offset = window_size//2
        height, width = image.shape
        R = np.zeros_like(image, dtype=np.float32)
        for y in range(offset, height-offset):
            for x in range(offset, width-offset):
                Sxx = np.sum(Ixx[y-offset:y+offset+1, x-offset:x+offset+1])
                Syy = np.sum(Iyy[y-offset:y+offset+1, x-offset:x+offset+1])
                Sxy = np.sum(Ixy[y-offset:y+offset+1, x-offset:x+offset+1])
                det = Sxx*Syy - Sxy**2
                trace = Sxx + Syy
                R[y, x] = det - k*(trace**2)
        return R

    def harris_corner_detection(self, image, R, alpha=0.1, window_size=3):
        offset = window_size//2
        height, width = image.shape
        threshold = alpha * R.max()
        cornerList = []
        for y in range(offset, height-offset):
            for x in range(offset, width-offset):
                value = R[y, x]
                if value > threshold:
                    cornerList.append([x, y, value])
        return cornerList

    def draw_corners(self, image, cornerList):
        corners_image = image.copy()
        for corner in cornerList:
            cv2.circle(corners_image, (corner[0], corner[1]), 4, (0, 255, 0))
        return corners_image

    def harris_corner_detection_main(self, image_original, image_copy, window_size=3, k=0.04, alpha=0.1):
        Ix, Iy = self.calculate_Ix_Iy(image_copy)
        Ixx, Iyy, Ixy = self.calculate_Ixx_Iyy_Ixy(Ix, Iy)
        R = self.calculate_matrix_M(image_copy, Ixx, Iyy, Ixy)
        cornerList = self.harris_corner_detection(image_copy, R, alpha)
        img_corners = self.draw_corners(image_original, cornerList)
        return img_corners

    def calculate_gradients(self):
        img = self.harris_img_input.img_copy
        img = cv2.GaussianBlur(img, (5, 5), 0)
        sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        ix = cv2.filter2D(img, -1, sobel, borderType=cv2.BORDER_REFLECT101)
        iy = cv2.filter2D(img, -1, sobel.T, borderType=cv2.BORDER_REFLECT101)
        return ix, iy

    def calculate_structure_tensor(self, ix, iy, window=3):
        ixx = cv2.blur(ix * ix, (window, window))
        iyy = cv2.blur(iy * iy, (window, window))
        ixy = cv2.blur(ix * iy, (window, window))
        return ixx, iyy, ixy

    def calculate_lambda_min(self, ixx, ixy, iyy, q=0.998):
        lambdamat = np.zeros_like(ixx)
        for x, y in np.ndindex(ixx.shape):
            H = np.array([[ixx[x, y], ixy[x, y]], [ixy[x, y], iyy[x, y]]])
            eigvals = np.linalg.eigvalsh(H)
            lambdamin = eigvals.min(initial=0)
            lambdamat[x, y] = lambdamin
        threshold = np.quantile(np.abs(lambdamat), q)
        lambdamat = np.abs(lambdamat) > threshold
        return lambdamat

    def lambdamin(self, image, window=3, q=0.998):
        ix, iy = Harris.calculate_gradients(self)
        ixx, iyy, ixy = Harris.calculate_structure_tensor(self, ix, iy, window)
        lambdamat = Harris.calculate_lambda_min(self, ixx, ixy, iyy, q)
        corners = np.argwhere(lambdamat)
        self.corners_image = self.harris_img_input.img_original.copy()
        for corner in corners:
            cv2.circle(self.corners_image,
                       (corner[1], corner[0]), 4, (0, 255, 0))
        return self.corners_image

    def lambdamin_main(self):
        img_corners = Harris.lambdamin(self, self.harris_img_input.img_copy)
        return img_corners


# corner_image = Harris.lambdamin_main(self)

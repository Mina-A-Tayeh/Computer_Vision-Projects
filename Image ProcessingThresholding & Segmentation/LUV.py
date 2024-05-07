import numpy as np
import cv2
from Image import Image


class LUV(Image):
    def __init__(self):
        super().__init__()

    def rgb_to_luv_image(rgb_image_path):
        # Conversion matrix from RGB to XYZ
        rgb_to_xyz_matrix = np.array([[0.412453, 0.357580, 0.180423],
                                      [0.212671, 0.715160, 0.072169],
                                      [0.019334, 0.119193, 0.950227]])

        # Define D65 white point and CIE chromaticity coordinates
        xn = 0.312713
        yn = 0.329016
        un = 4 * xn / (-2 * xn + 12 * yn + 3)
        vn = 9 * yn / (-2 * xn + 12 * yn + 3)

        def xyz_to_luv(xyz_image):
            # Normalize XYZ values
            xyz_normalized = xyz_image / np.array([0.95047, 1.00000, 1.08883])

            # Compute intermediate values
            L = np.where(xyz_normalized[..., 1] > 0.008856,
                         116 * np.power(xyz_normalized[..., 1], 1/3) - 16,
                         903.3 * xyz_normalized[..., 1])
            eps = np.finfo(float).eps  # Smallest positive float epsilon
            u_denom = xyz_normalized[..., 0] + 15 * \
                xyz_normalized[..., 1] + 3 * xyz_normalized[..., 2]
            v_denom = xyz_normalized[..., 0] + 15 * \
                xyz_normalized[..., 1] + 3 * xyz_normalized[..., 2]
            u = 13 * L * np.where(u_denom > eps,
                                  (4 * xyz_normalized[..., 0] / u_denom) - un, 0)
            v = 13 * L * np.where(v_denom > eps,
                                  (9 * xyz_normalized[..., 1] / v_denom) - vn, 0)

            luv_image = np.stack([L, u, v], axis=-1)
            luv_image[..., 0] = np.clip(
                luv_image[..., 0] / 100.0, 0, 1)  # Normalize L component
            luv_image[..., 1:] = (luv_image[..., 1:] + np.array([134, 140])) / \
                np.array([354, 256])  # Normalize U and V components
            return luv_image

        def rgb_to_xyz(rgb_image):
            return np.dot(rgb_image.reshape(-1, 3), rgb_to_xyz_matrix.T).reshape(rgb_image.shape)

        def rgb_to_luv(rgb_image_path):
            # Load the RGB image
            rgb_image = cv2.imread(rgb_image_path)

            # Convert BGR to RGB
            rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            # Convert RGB to XYZ
            xyz_image = rgb_to_xyz(rgb_image_rgb)

            # Convert XYZ to LUV
            luv_image = xyz_to_luv(xyz_image)

            return rgb_image_rgb, luv_image

        return rgb_to_luv(rgb_image_path)

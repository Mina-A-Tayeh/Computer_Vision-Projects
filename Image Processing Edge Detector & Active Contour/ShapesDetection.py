from Image import Image
import numpy as np
import cv2
import scipy.ndimage
from scipy.ndimage import convolve
from collections import defaultdict


class ShapesDetection(Image):
    def __init__(self):
        super().__init__()

    @staticmethod
    def detect_and_draw_lines(image, theta_res=1, rho_res=1, num_peaks=25, threshold=50, nhood_size=3, color=(0, 255, 0)):
        """
        Detect and draw lines on an image using Hough Line Transform.

        Args:
        - image: Input image.
        - theta_res: Resolution of theta in degrees.
        - rho_res: Resolution of rho.
        - num_peaks: Number of peaks (lines) to detect.
        - threshold: Threshold value to consider a peak.
        - nhood_size: Size of the neighborhood to suppress around each peak.
        - color: Color of the detected lines.

        Returns:
        - image_with_lines: Image with detected lines drawn on it.
        """

        # Convert image to grayscale if necessary
        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Edge detection using Canny or any other edge detection method
        edges = cv2.Canny(gray_image, 50, 150)

        # Hough Line Transform
        diagonal = int(np.sqrt(edges.shape[0] ** 2 + edges.shape[1] ** 2))
        thetas = np.deg2rad(np.arange(0, 180, theta_res))
        rhos = np.arange(-diagonal, diagonal + 1, rho_res)
        H = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)

        y_idxs, x_idxs = np.nonzero(edges)
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]
            for theta_idx, theta in enumerate(thetas):
                rho = int(round(x * np.cos(theta) +
                                y * np.sin(theta))) + diagonal
                H[rho, theta_idx] += 1

        # Find peaks in the Hough accumulator matrix
        peaks = []
        for _ in range(num_peaks):
            idx = np.argmax(H)
            rho_idx, theta_idx = np.unravel_index(idx, H.shape)

            if H[rho_idx, theta_idx] >= threshold:
                peaks.append((rho_idx, theta_idx))

                # Suppress the neighborhood of the peak
                y_min = max(0, rho_idx - nhood_size // 2)
                y_max = min(H.shape[0], rho_idx + nhood_size // 2 + 1)
                x_min = max(0, theta_idx - nhood_size // 2)
                x_max = min(H.shape[1], theta_idx + nhood_size // 2 + 1)
                H[y_min:y_max, x_min:x_max] = 0

        # Convert peaks to (rho, theta) pairs
        lines = [(rhos[rho_idx], thetas[theta_idx])
                 for rho_idx, theta_idx in peaks]

        # Draw lines on the original image
        image_with_lines = image.copy()
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image_with_lines, (x1, y1), (x2, y2), color, 2)

        return image_with_lines

    @staticmethod
    def detect_and_draw_hough_circles(image, threshold=8.1, region=15, radius=[70, 10]):
        """
        Detect circles in the input image using Hough Transform and draw them.

        Args:
        - image (numpy.ndarray): Input image.
        - threshold (float): Threshold for circle detection.
        - region (int): Region size for local maximum search.
        - radius (list): Range of radii to search for circles.

        Returns:
        - numpy.ndarray: Processed image with detected circles drawn.

        The function first applies Gaussian smoothing and Canny edge detection to the input image.
        It then constructs an accumulator array to detect circles of different radii.
        The accumulator array is initialized with zeros, and each edge point contributes to the
        accumulator by drawing circles of different radii centered at that point.
        Local maxima in the accumulator array indicate potential circle centers, and the function
        extracts these centers and draws circles on the input image.

        Note:
        - The input image should be in BGR format.
        - The returned image is in RGB format with detected circles drawn on it.
        """

        # Make a copy of the input image to prevent modifications to the original
        modified_image = np.copy(image)

        # Convert image to grayscale if it's in color
        if len(modified_image.shape) > 2:
            grayscale_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale_image = modified_image

        # Apply Gaussian filter for smoothing
        smoothed_image = cv2.GaussianBlur(grayscale_image, (7, 7), 2)

        # Apply Canny edge detection
        edges = cv2.Canny(smoothed_image, 50, 128)

        # Get image dimensions
        (height, width) = edges.shape

        # Determine maximum and minimum radii
        if radius is None:
            max_radius = max(height, width)
            min_radius = 3
        else:
            [max_radius, min_radius] = radius

        num_radii = max_radius - min_radius

        # Initialize accumulator array
        accumulator = np.zeros(
            (max_radius, height + 2 * max_radius, width + 2 * max_radius))
        detected_circles = np.zeros(
            (max_radius, height + 2 * max_radius, width + 2 * max_radius))

        # Precompute angles
        angles = np.arange(0, 360) * np.pi / 180
        edges_coordinates = np.argwhere(edges)

        # Iterate over radii
        for r_idx in range(num_radii):
            radius = min_radius + r_idx

            # Create circle template
            circle_template = np.zeros((2 * (radius + 1), 2 * (radius + 1)))
            # Center of the circle template
            (center_x, center_y) = (radius + 1, radius + 1)
            for angle in angles:
                x = int(np.round(radius * np.cos(angle)))
                y = int(np.round(radius * np.sin(angle)))
                circle_template[center_x + x, center_y + y] = 1

            template_size = np.argwhere(circle_template).shape[0]

            # Iterate over edge points
            for x, y in edges_coordinates:
                # Center the circle template over the edge point and update the accumulator array
                X = [x - center_x + max_radius, x + center_x + max_radius]
                Y = [y - center_y + max_radius, y + center_y + max_radius]
                accumulator[radius, X[0]:X[1], Y[0]:Y[1]] += circle_template

            accumulator[radius][accumulator[radius] <
                                threshold * template_size / radius] = 0

        # Find local maxima in the accumulator array
        for r, x, y in np.argwhere(accumulator):
            local_maxima = accumulator[r - region:r + region,
                                       x - region:x + region, y - region:y + region]
            try:
                p, a, b = np.unravel_index(
                    np.argmax(local_maxima), local_maxima.shape)
            except:
                continue
            detected_circles[r + (p - region), x +
                             (a - region), y + (b - region)] = 1

        # Extract circle information and draw circles on the image
        circle_coordinates = np.argwhere(detected_circles)
        for r, x, y in circle_coordinates:
            cv2.circle(modified_image, (y - max_radius,
                       x - max_radius), r, (0, 0, 255), 2)

        # Convert BGR image to RGB format
        modified_image_rgb = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)

        return modified_image_rgb

    @staticmethod
    def detect_and_draw_hough_ellipses(image, a_min=30, a_max=100, b_min=30, b_max=100, delta_a=2, delta_b=2, num_thetas=100, bin_threshold=0.4, min_edge_threshold=100, max_edge_threshold=150):
        """
        Detect ellipses using Hough Transform.
        Args:
            image_path (str): Path to the input image file.
            a_min (int): Minimum semi-major axis length of ellipses to detect.
            a_max (int): Maximum semi-major axis length of ellipses to detect.
            b_min (int): Minimum semi-minor axis length of ellipses to detect.
            b_max (int): Maximum semi-minor axis length of ellipses to detect.
            delta_a (int): Step size for semi-major axis length.
            delta_b (int): Step size for semi-minor axis length.
            num_thetas (int): Number of steps for theta from 0 to 2PI.
            bin_threshold (float): Thresholding value in percentage to shortlist candidate ellipses.
            min_edge_threshold (int): Minimum threshold value for edge detection.
            max_edge_threshold (int): Maximum threshold value for edge detection.
        Returns:
            tuple: A tuple containing the output image with detected ellipses drawn and a list of detected ellipses.
        """

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edge_image = cv2.Canny(
            gray_image, min_edge_threshold, max_edge_threshold)

        # Get image dimensions
        img_height, img_width = edge_image.shape[:2]

        # Initialize parameters for Hough ellipse detection
        dtheta = int(360 / num_thetas)
        thetas = np.arange(0, 360, step=dtheta)
        as_ = np.arange(a_min, a_max, step=delta_a)
        bs = np.arange(b_min, b_max, step=delta_b)
        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))
        ellipse_candidates = [(a, b, int(a * cos_thetas[t]), int(b * sin_thetas[t]))
                              for a in as_ for b in bs for t in range(num_thetas)]

        # Initialize accumulator
        accumulator = defaultdict(int)

        # Iterate over each pixel and vote for potential ellipse centers
        for y in range(img_height):
            for x in range(img_width):
                if edge_image[y][x] != 0:
                    for a, b, acos_t, bsin_t in ellipse_candidates:
                        x_center = x - acos_t
                        y_center = y - bsin_t
                        accumulator[(x_center, y_center, a, b)] += 1

        # Initialize output image
        output_img = image.copy()

        # Store detected ellipses
        out_ellipses = []

        # Loop through the accumulator to find ellipses based on the threshold
        for candidate_ellipse, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
            x, y, a, b = candidate_ellipse
            current_vote_percentage = votes / num_thetas
            if current_vote_percentage >= bin_threshold:
                out_ellipses.append((x, y, a, b, current_vote_percentage))

        # Perform post-processing to remove duplicate ellipses
        pixel_threshold = 5
        postprocess_ellipses = []
        for x, y, a, b, v in out_ellipses:
            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(a - ac) > pixel_threshold or abs(b - bc) > pixel_threshold for xc, yc, ac, bc, v in postprocess_ellipses):
                postprocess_ellipses.append((x, y, a, b, v))
        out_ellipses = postprocess_ellipses

        # Draw detected ellipses on the output image
        for x, y, a, b, v in out_ellipses:
            output_img = cv2.ellipse(
                output_img, (x, y), (a, b), 0, 0, 360, (0, 255, 0), 2)

        return output_img, out_ellipses

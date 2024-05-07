import numpy as np
import cv2
from Image import Image
# from google.colab.patches import cv2_imshow


class ActiveContour(Image):
    def __init__(self):
        super().__init__()
        self.center = (400, 350)
        self.radius = 100
        self.numOfIterations = 100
        self.alpha = 9
        self.beta = 9
        self.gamma = 1

    def initial_contour(self, center, radius):
        initial_snake = []
        current_angle = 0
        resolution = 360 / 1000.0
        for i in range(1000):
            # Ensure angle has dtype=np.float32
            angle = np.array([current_angle], dtype=np.float64)
            x, y = cv2.polarToCart(
                np.array([radius], dtype=np.float64), angle, True)

            # Access the first elements of x and y arrays
            y_point = int(y[0][0] + center[1])
            x_point = int(x[0][0] + center[0])

            current_angle += resolution
            initial_snake.append((x_point, y_point))
        return initial_snake

    def calcInternalEnergy(self, pt, prevPt, nextPt, alpha):
        dx1 = pt[0] - prevPt[0]
        dy1 = pt[1] - prevPt[1]
        dx2 = nextPt[0] - pt[0]
        dy2 = nextPt[1] - pt[1]

        denominator = pow(dx1*dx1 + dy1*dy1, 1.5)
        if denominator == 0:
            return 0  # Handle the case when denominator is zero

        curvature = (dx1 * dy2 - dx2 * dy1) / denominator
        return alpha * curvature

    def calcExternalEnergy(self, img, pt, beta):
        # Assuming higher intensity for stronger edges
        return -beta * img[pt[1], pt[0]]

    def calcBalloonEnergy(self, pt, prevPt, gamma):
        dx = pt[0] - prevPt[0]
        dy = pt[1] - prevPt[1]
        return gamma * (dx*dx + dy*dy)

    def contourUpdating(self):
        snake_points = self.initial_contour(self.center, self.radius)
        # grayImg = cv2.cvtColor(self.img_copy, cv2.COLOR_BGR2GRAY)
        grayImg = self.img_copy
        grayImg = cv2.blur(grayImg, (5, 5))

        for _ in range(self.numOfIterations):
            numPoints = len(snake_points)
            newCurve = [None] * numPoints

            for i in range(numPoints):
                pt = snake_points[i]
                prevPt = snake_points[(i - 1 + numPoints) % numPoints]
                nextPt = snake_points[(i + 1) % numPoints]
                minEnergy = float('inf')
                newPt = pt

                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        movePt = (pt[0] + dx, pt[1] + dy)
                        internal_e = self.calcInternalEnergy(
                            movePt, prevPt, nextPt, self.alpha)
                        external_e = self.calcExternalEnergy(
                            grayImg, movePt, self.beta)
                        balloon_e = self.calcBalloonEnergy(
                            movePt, prevPt, self.gamma)
                        energy = internal_e + external_e + balloon_e

                        if energy < minEnergy:
                            minEnergy = energy
                            newPt = movePt

                newCurve[i] = newPt

            snake_points = newCurve

        perimeter = 0
        prevDir = 0
        for currPt, prevPt in zip(snake_points, snake_points[:-1] + [snake_points[0]]):
            dx = currPt[0] - prevPt[0]
            dy = currPt[1] - prevPt[1]

            dir = 0  # Initialize dir before any conditional assignment
            # Map directions to integer codes (adjust based on your chain code convention)
            if dx == 0 and dy == 1:
                dir = 0
            elif dx == -1 and dy == 1:
                dir = 1
            elif dx == -1 and dy == 0:
                dir = 2
            elif dx == -1 and dy == -1:
                dir = 3
            elif dx == 0 and dy == -1:
                dir = 4
            elif dx == 1 and dy == -1:
                dir = 5
            elif dx == 1 and dy == 0:
                dir = 6
            elif dx == 1 and dy == 1:
                dir = 7

            dir = (dir - prevDir + 8) % 8  # Calculation is always performed
            perimeter += np.sqrt(dx**2 + dy**2)
            prevDir = dir

        # Calculate area using polygonal approximation (for 8-direction chain code)
        approx_polygon = np.array(snake_points)
        area = cv2.contourArea(approx_polygon)

        return snake_points, perimeter, area

    def display_output_contour(self, snake_contour, output_viewer):
        # Draw the initial and final snake contours on the image
        output_img = self.img_original.copy()
        cv2.circle(output_img, self.center, self.radius,
                   (0, 255, 0), 2)  # Draw initial circle
        for i in range(len(snake_contour)):
            cv2.circle(
                output_img, snake_contour[i], 4, (0, 0, 255), thickness=1)
            # Draw line between current and next point (cyclic connection)
            next_i = (i + 1) % len(snake_contour)
            # blue for connection
            cv2.line(output_img, snake_contour[i],
                     snake_contour[next_i], (0, 0, 255), 1)

        cv2.line(output_img, snake_contour[0],
                 snake_contour[-1], (0, 0, 255), 1)

        output_viewer.display_image(output_img, False)

    def display_initial_contour(self, output_viewer):
        output_img = self.img_original.copy()
        cv2.circle(output_img, self.center, self.radius,
                   (0, 255, 0), 2)
        output_viewer.display_image(output_img, False)

from Image import Image
import numpy as np
import cv2


#regionGrow algorithm
class RegionGrowing(Image):
    def __init__(self):
        super().__init__()

    def get_difference(self, img, current_point, point_2):
        x1, y1 = current_point
        x2, y2 = point_2
        return abs(int(img[x1, y1]) - int(img[x2, y2]))

    def get_around_pixels(self):
        around = [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)]
        return around

    def region_grow_segmentation(self, img, seeds, threshold):
        height, weight = img.shape
        seeds_array = np.zeros(img.shape)

        label = 1
        around_pixels = self.get_around_pixels()

        while len(seeds) > 0:
            current_point = seeds.pop(0)

            seeds_array[current_point[0], current_point[1]] = label

            for i in range(8):
                neighbor_x = current_point[0] + around_pixels[i][0]
                neighbor_y = current_point[1] + around_pixels[i][1]
                if neighbor_x < 0 or neighbor_y < 0 or neighbor_x >= height or neighbor_y >= weight:
                    continue
                grey_difference = self.get_difference(img, current_point, (neighbor_x, neighbor_y))
                if grey_difference < threshold and seeds_array[neighbor_x, neighbor_y] == 0:
                    seeds_array[neighbor_x, neighbor_y] = label
                    seeds.append((neighbor_x, neighbor_y))
        return seeds_array

    def assign_seed(self, seeds,seeds_array):
        for seed in seeds_array:
            print(f"seed: {seed}")
            x, y = seed
            print(f"x: {x}, y: {y}")
            seeds.append((x, y))

    def apply_region_growing(self, source: np.ndarray,threshold,seeds_array):
        print(f"seeds_array: {seeds_array}")
        src = np.copy(source)
        color_img = cv2.cvtColor(src, cv2.COLOR_LUV2BGR)
        img_gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        seeds = []
        self.assign_seed(seeds,seeds_array)
        output_image = self.region_grow_segmentation(img_gray, seeds, threshold)

        return output_image

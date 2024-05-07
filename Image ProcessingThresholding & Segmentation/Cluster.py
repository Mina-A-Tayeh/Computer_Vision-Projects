from Image import Image
import cv2
import numpy as np
import random
from scipy.spatial import KDTree



class Cluster(Image):
    def __init__(self):
        super().__init__()
        self.methods = {"K means": self.k_means,
                        "Agglomerative": self.apply_agglomerative_clustering,
                        "Mean Shift": self.mean_shift
                        }

######################################################## k means ########################################################

    def k_means(self, image, k):
        centroids = self.initialize_centroids(image, k)
        for _ in range(10):  # Perform 10 iterations
            clusters = self.build_clusters(image, centroids)
            new_centroids = self.adjust_centroids(image, clusters)
            if np.allclose(centroids, new_centroids):
                break  # If centroids don't change, exit loop
            centroids = new_centroids
        segmented_image = self.paint_image(image, clusters)
        return segmented_image

    def initialize_centroids(self, image, k):
        random.seed(cv2.getTickCount())
        rows, cols, _ = image.shape
        centroids = []
        for _ in range(k):
            centroid = (random.randint(0, cols - 1),
                        random.randint(0, rows - 1))
            centroids.append(image[centroid[1], centroid[0]])
        return centroids

    def get_euclidean_distance(self, pixel, centroid):
        return np.linalg.norm(pixel - centroid)

    def build_clusters(self, image, centroids):
        clusters = [[] for _ in range(len(centroids))]
        rows, cols, _ = image.shape
        for row in range(rows):
            for col in range(cols):
                pixel = image[row, col]
                min_distance = float('inf')
                closest_cluster_index = 0
                for i, centroid in enumerate(centroids):
                    distance = self.get_euclidean_distance(pixel, centroid)
                    if distance < min_distance:
                        min_distance = distance
                        closest_cluster_index = i
                clusters[closest_cluster_index].append((col, row))
        return clusters

    def adjust_centroids(self, image, clusters):
        new_centroids = []
        for cluster in clusters:
            if not cluster:
                # If cluster is empty, retain the previous centroid
                continue
            pixels = np.array([image[row, col] for col, row in cluster])
            new_centroid = np.mean(pixels, axis=0)
            new_centroids.append(new_centroid)
        return new_centroids

    def paint_image(self, image, clusters):
        segmented_image = np.zeros_like(image)
        for i, cluster in enumerate(clusters):
            # Extract color from original image
            color = np.mean([image[row, col] for col, row in cluster], axis=0)
            for col, row in cluster:
                segmented_image[row, col] = color
        return segmented_image

######################################################## Agglomerative ########################################################

    def initial_clusters(self, image_clusters, k):
        """
        Initializes the clusters with k colors by grouping the image pixels based on their color.
        """
        # Determine the range of colors for each cluster
        cluster_color = int(256 / k)
        # Initialize empty groups for each cluster
        groups = [[] for _ in range(k)]

        # Assign each pixel to its closest cluster based on color
        for p in image_clusters:
            # Calculate the mean color of the pixel
            color = int(np.mean(p))
            # Determine the index of the closest cluster
            group_index = min(range(k), key=lambda i: abs(
                color - i * cluster_color))
            # Add the pixel to the corresponding cluster
            groups[group_index].append(p)

        # Remove empty clusters if any
        return [group for group in groups if group]

    def get_cluster_center(self, cluster):
        """
        Returns the center of the cluster.
        """
        # Calculate the mean of all points in the cluster
        return np.mean(cluster, axis=0)

    def get_clusters(self, image_clusters, clusters_number):
        """
        Agglomerative clustering algorithm to group the image pixels into a specified number of clusters.
        """
        # Initialize clusters and their assignments
        clusters_list = self.initial_clusters(image_clusters, clusters_number)
        cluster_assignments = {tuple(point): i for i, cluster in enumerate(
            clusters_list) for point in cluster}
        # Calculate initial cluster centers
        centers = [self.get_cluster_center(cluster)
                   for cluster in clusters_list]

        # Merge clusters until the desired number is reached
        while len(clusters_list) > clusters_number:
            min_distance = float('inf')
            merge_indices = None

            # Find the two clusters with the minimum distance
            for i, cluster1 in enumerate(clusters_list):
                for j, cluster2 in enumerate(clusters_list[:i]):
                    distance = self.get_euclidean_distance(
                        centers[i], centers[j])
                    if distance < min_distance:
                        min_distance = distance
                        merge_indices = (i, j)

            # Merge the closest clusters
            i, j = merge_indices
            clusters_list[i] += clusters_list[j]
            del clusters_list[j]
            # Update cluster centers
            centers[i] = self.get_cluster_center(clusters_list[i])
            del centers[j]

            # Update cluster assignments
            for point in clusters_list[i]:
                cluster_assignments[tuple(point)] = i

        return cluster_assignments, centers

    def apply_agglomerative_clustering(self, image, clusters_number):
        """
        Applies agglomerative clustering to the image and returns the segmented image.
        """
        # Reshape the image for processing
        flattened_image = image.reshape((-1, 3))
        # Perform agglomerative clustering
        cluster_assignments, centers = self.get_clusters(
            flattened_image, clusters_number)
        # Assign each pixel in the image to its corresponding cluster center
        output_image = np.array([centers[cluster_assignments[tuple(p)]]
                                for p in flattened_image], dtype=np.uint8)
        # Reshape the segmented image to its original shape
        output_image = output_image.reshape(image.shape)
        return output_image

    ######################################################## Mean Shift ########################################################

    def mean_shift(self, image, bandwidth=30,criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)):

        """ Applies mean shift clustering to the image and returns the segmented image. """

        flattened_image = image.reshape((-1, 3))

        num_points, num_features = flattened_image.shape
        point_considered = np.zeros(num_points, dtype=bool)
        labels = -1 * np.ones(num_points, dtype=int)
        label_count = 0

        tree = KDTree(flattened_image)

        for i in range(num_points):
            if point_considered[i]:
                continue

            Center_point = flattened_image[i]
            while True:
                in_window = tree.query_ball_point(Center_point, r=bandwidth)
                new_center = np.mean(flattened_image[in_window], axis=0)

                if np.linalg.norm(new_center - Center_point) < criteria[0]:
                    labels[in_window] = label_count
                    point_considered[in_window] = True
                    label_count += 1
                    break

                Center_point = new_center

        # Generate a unique color for each label
        unique_colors = np.random.randint(0, 255, (label_count, 3))

         # Create a new image where each pixel is assigned the color of its cluster
        new_img = np.zeros_like(image)
        for i in range(label_count):
            new_img[labels.reshape(image.shape[:2]) == i] = unique_colors[i]


        output_image = np.array(new_img, np.uint8)
    

        return output_image

    ######################################################## Event handler ########################################################

    def apply_clustering(self, image, method, parameter):
        return self.methods[method](image, parameter)

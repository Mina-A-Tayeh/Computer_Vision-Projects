from RegionGrowing import RegionGrowing
from Thresholder import Thresholder
from Cluster import Cluster


def init_ui(self):
    region_growing_tab(self)
    threshold_tab(self)
    clustering_tab(self)


def region_growing_tab(self):
    self.region_growing_img_input = RegionGrowing()
    self.region_growing_img_output = RegionGrowing()
    region_growing_images = [
        self.region_growing_img_input, self.region_growing_img_output]
    region_growing_labels = ["Input Image", "Output Image"]
    self.add_image_viewers(self.region_growing_horizontal_layout,
                           region_growing_images, region_growing_labels)
    self.actionOpen.triggered.connect(
        lambda: self.upload_image(self.region_growing_img_input))
    self.region_growing_img_input.mousePressEvent = lambda event: self.mousePressEvent(
        event, self.region_growing_img_input)


def threshold_tab(self):
    self.threshold_img_input = Thresholder()
    self.threshold_img_output = Thresholder()
    threshold_images = [self.threshold_img_input, self.threshold_img_output]
    threshold_labels = ["Input Image", "Output Image"]
    self.add_image_viewers(self.threshold_layout,
                           threshold_images, threshold_labels)

    self.threshold_img_input.mouseDoubleClickEvent = lambda event: self.mouseDoubleClickEvent(
        event, self.threshold_img_input)

    self.thresholding_types.currentIndexChanged.connect(
        self.on_combo_box_changed)

    self.global_thresholding.clicked.connect(
        lambda: self.on_radio_button_clicked(self.global_thresholding))
    self.local_thresolding.clicked.connect(
        lambda: self.on_radio_button_clicked(self.local_thresolding))

    self.thresholding_apply.clicked.connect(
        lambda: self.apply_thersholding(self.threshold_img_input, self.threshold_img_output))


def clustering_tab(self):
    self.clustering_img_input = Cluster()
    self.clustering_img_output = Cluster()
    clustering_images = [self.clustering_img_input, self.clustering_img_output]
    clustering_labels = ["Input Image", "Output Image"]
    self.add_image_viewers(self.clustering_layout,
                           clustering_images, clustering_labels)

    self.clustering_img_input.mouseDoubleClickEvent = lambda event: self.mouseDoubleClickEvent(
        event, self.clustering_img_input)

    self.clustering_apply.clicked.connect(
        lambda: self.apply_clustering(self.clustering_img_input, self.clustering_img_output))

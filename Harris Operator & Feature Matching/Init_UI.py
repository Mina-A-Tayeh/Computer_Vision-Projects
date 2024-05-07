from Harris import Harris
from Sift import Sift
from Matcher import Matcher


def init_ui(self):
    harris_tab(self)
    sift_tab(self)
    matching_tab(self)


def harris_tab(self):
    self.harris_img_input = Harris()
    self.harris_img_output = Harris()
    harris_images = [self.harris_img_input, self.harris_img_output]
    harris_labels = ["Input Image", "Output Image"]
    self.add_image_viewers(self.horizontalLayout1,
                           harris_images, harris_labels)
    self.Alpha_slider.setValue(1)
    # set ranges for sliders
    self.Alpha_slider.setRange(1, 100)
    # set aplha when slider is moved
    self.Alpha_slider.valueChanged.connect(
        lambda: self.alpha_label.setText(f"Alpha: {self.Alpha_slider.value()/100}"))
    self.harris_img_input.mouseDoubleClickEvent = lambda event: self.mouseDoubleClickEvent(
        event, self.harris_img_input)

    self.harris_apply.clicked.connect(
        lambda: self.apply_harris(self.harris_img_input, self.harris_img_output))

    self.lambdamin_apply.clicked.connect(self.apply_lambdamin)


def sift_tab(self):
    self.sift_img_input = Sift()
    self.sift_img_output = Sift()
    sift_images = [self.sift_img_input, self.sift_img_output]
    sift_labels = ["Input Image", "Output Image"]
    self.add_image_viewers(self.sift_layout,
                           sift_images, sift_labels)
    self.sift_img_input.mouseDoubleClickEvent = lambda event: self.descriptor_tab(
        event, self.sift_img_input)

    self.sift_apply.clicked.connect(
        lambda: self.apply_sift(self.sift_img_input, self.sift_img_output))


def matching_tab(self):
    self.matching_img_input1 = Matcher()
    self.matching_img_input2 = Matcher()
    self.matching_img_output = Matcher()
    matching_images = [self.matching_img_input1,
                       self.matching_img_input2]
    matching_labels = ["Input Image", "Input Image"]
    self.add_image_viewers(self.feature_input_vlayout,
                           matching_images, matching_labels)
    self.add_image_viewers(self.feature_output_layout,
                           [self.matching_img_output], ["Output Image"])
    self.matching_img_input1.mouseDoubleClickEvent = lambda event: self.mouseDoubleClickEvent(
        event, self.matching_img_input1)
    self.matching_img_input2.mouseDoubleClickEvent = lambda event: self.mouseDoubleClickEvent(
        event, self.matching_img_input2)

    self.apply_feature.clicked.connect(
        lambda: self.apply_matching(self.matching_img_input1, self.matching_img_input2, self.matching_img_output))

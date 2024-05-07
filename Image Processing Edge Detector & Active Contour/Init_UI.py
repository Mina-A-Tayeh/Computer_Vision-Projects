from Canny import *
from ShapesDetection import *
from ActiveContour import *


def init_ui(self):
    hough_tab(self)
    shape_detector(self)
    activecontour(self)


def hough_tab(self):
    self.hough_img_input = Canny()
    self.hough_img_output = Canny()
    canny_images = [self.hough_img_input, self.hough_img_output]
    canny_labels = ["Input Image", "Output Image"]
    self.add_image_viewers(self.horizontalLayout1, canny_images, canny_labels)
    self.hough_img_input.mouseDoubleClickEvent = lambda event: self.canny_tab(
        event, self.hough_img_input)
    # when button clicked apply canny edge detection

    self.canny_apply_button.clicked.connect(self.canny_slider_changed)
    self.sigma_slider.valueChanged.connect(self.update_canny_sliders)
    self.threshold1_slider.valueChanged.connect(self.update_canny_sliders)
    self.threshold2_slider.valueChanged.connect(self.update_canny_sliders)


def shape_detector(self):
    self.shape_detection_img_input = ShapesDetection()
    self.shape_detection_img_output = ShapesDetection()
    shape_detection_images = [
        self.shape_detection_img_input, self.shape_detection_img_output]
    shape_detection_labels = ["Input Image", "Output Image"]
    self.add_image_viewers(self.horizontalLayout2,
                           shape_detection_images, shape_detection_labels)
    self.shape_detection_img_input.mouseDoubleClickEvent = lambda event: self.shape_detection_tab(
        event, self.shape_detection_img_input)

    self.shape_detection_apply_button.clicked.connect(
        self.shape_detection_apply)


def activecontour(self):
    self.active_contour_img_input = ActiveContour()
    self.active_contour_img_output = ActiveContour()
    active_contour_images = [
        self.active_contour_img_input, self.active_contour_img_output]
    active_contour_labels = ["Input Image", "Output Image"]
    self.add_image_viewers(self.horizontalLayout1_2,
                           active_contour_images, active_contour_labels)
    self.active_contour_img_input.mouseDoubleClickEvent = lambda event: self.mouseDoubleClickEvent(
        event, self.active_contour_img_input)

    self.active_contour_apply.clicked.connect(
        self.apply_active_contour)
    self.x_spinbox.valueChanged.connect(self.draw_init_circle)
    self.y_spinbox.valueChanged.connect(self.draw_init_circle)
    self.r_spinbox.valueChanged.connect(self.draw_init_circle)

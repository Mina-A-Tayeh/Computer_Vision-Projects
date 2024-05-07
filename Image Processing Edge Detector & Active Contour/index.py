from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from os import path
import sys
from Image import *
from Init_UI import *

FORM_CLASS, _ = loadUiType(
    path.join(path.dirname(__file__), "Task2.ui"))


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setGeometry(0, 0, 1920, 1080)
        self.setWindowTitle("Image Viewer")
        self.setupUi(self)
        init_ui(self)

    def mouseDoubleClickEvent(self, event, img):
        if event.button() == Qt.LeftButton:
            img.read_image()

    def add_image_viewers(self, h_layout, images, images_labels):
        for i in range(len(images)):
            group_box = QGroupBox(images_labels[i])
            group_box_layout = QVBoxLayout(group_box)
            group_box_layout.addWidget(images[i])
            h_layout.addWidget(group_box)

# canny tab
    def canny_tab(self, event, img):
        # set range for the sigma slider from 0 to 5
        self.mouseDoubleClickEvent(event, img)
        self.sigma_slider.setRange(0, 5)
        self.threshold1_slider.setRange(0, 100)
        self.threshold2_slider.setRange(0, 100)

    def canny_slider_changed(self):
        sigma_value = self.sigma_slider.value()
        low_threshold_value = (self.threshold1_slider.value())/100
        high_threshold_value = (self.threshold2_slider.value())/100
        # Create an instance of the Image class
        gaussian_image = Canny.canny_edge_detector(
            self, sigma=sigma_value, low_threshold_ratio=low_threshold_value, high_threshold_ratio=high_threshold_value)
        self.hough_img_output.display_image(gaussian_image)

    def update_canny_sliders(self):
        sigma_value = self.sigma_slider.value()
        low_threshold_value = self.threshold1_slider.value()
        high_threshold_value = self.threshold2_slider.value()
        self.sigma_lcd.display(sigma_value)
        self.threshold1_lcd.display(low_threshold_value/100)
        self.threshold2_lcd.display(high_threshold_value/100)

    # shape detection tab
    def shape_detection_tab(self, event, img):
        self.mouseDoubleClickEvent(event, img)

    def shape_detection_apply(self):
        curr_mode = self.mode.currentIndex()
        self.shape_detection_img_input.img_copy = self.shape_detection_img_input.img_original
        # Create an instance of the Image class
        if curr_mode == 0:
            out_image = ShapesDetection.detect_and_draw_lines(
                image=self.shape_detection_img_input.img_copy)
        elif curr_mode == 1:
            out_image = ShapesDetection.detect_and_draw_hough_circles(
                image=self.shape_detection_img_input.img_copy)
        else:
            out_image, _ = ShapesDetection.detect_and_draw_hough_ellipses(
                image=self.shape_detection_img_input.img_copy)

        self.shape_detection_img_output.display_image(out_image)

    # active contour tab
    def apply_active_contour(self):
        input_img = self.active_contour_img_input
        output_img = self.active_contour_img_output

        x_coordinate = self.x_spinbox.value()
        y_coordinate = self.y_spinbox.value()
        radius = self.r_spinbox.value()
        alpha = self.alpha_spinbox.value()
        beta = self.beta_spinbox.value()
        gamma = self.gamma_spinbox.value()
        iterations = self.it_spinbox.value()

        input_img.center = (x_coordinate, y_coordinate)
        input_img.radius = radius
        input_img.alpha = alpha
        input_img.beta = beta
        input_img.gamma = gamma
        input_img.numOfIterations = iterations
        snake_points , perimeter , area = input_img.contourUpdating()
        input_img.display_output_contour(
                    snake_points, output_img)
        self.label_11.setText(f"Perimeter = {perimeter:.2f}")
        self.label_12.setText(f"Area = {area:.2f}")

    def draw_init_circle(self):
        input_img = self.active_contour_img_input
        output_img = self.active_contour_img_output
        x_coordinate = self.x_spinbox.value()
        y_coordinate = self.y_spinbox.value()
        radius = self.r_spinbox.value()
        input_img.center = (x_coordinate, y_coordinate)
        input_img.radius = radius
        input_img.display_initial_contour(output_img)

    def exit_program(self):
        sys.exit()


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

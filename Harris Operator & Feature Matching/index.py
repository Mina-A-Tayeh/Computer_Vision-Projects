from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from os import path
import sys
from Init_UI import *

FORM_CLASS, _ = loadUiType(
    path.join(path.dirname(__file__), "object_detection.ui"
              ))


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

    def corner_tab(self, event, img):
        self.mouseDoubleClickEvent(event, img)

    def descriptor_tab(self, event, img):
        self.mouseDoubleClickEvent(event, img)

    def apply_harris(self, image_input, image_output):
        alpha_value = self.Alpha_slider.value()/100
        corner_image = image_input.harris_corner_detection_main(
            image_input.img_original, image_input.img_copy, alpha=alpha_value)
        image_output.display_image(corner_image)

    def apply_lambdamin(self):
        corner_image = Harris.lambdamin_main(self)
        self.harris_img_output.display_image(corner_image)

    def apply_sift(self, image_input, image_output):
        # reset the image copy
        selected_keypoints_with_orientations, descriptors, keypoints_in_cv_format = image_input.detect_and_describe_features(
            image_input.img_copy)
        sift_image = image_input.display_keypoints_with_orientations(
            image_input.img_copy, keypoints_in_cv_format)
        # print(keypoints_in_cv_format)
        image_output.display_image(sift_image)

    def apply_matching(self, img1, img2, img_output):
        img_output.plot_features(img1, img2)

    def exit_program(self):
        sys.exit()


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

# alii

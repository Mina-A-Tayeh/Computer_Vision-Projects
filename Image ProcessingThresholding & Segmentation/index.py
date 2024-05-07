from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from os import path
import sys
from Init_UI import *

FORM_CLASS, _ = loadUiType(
    path.join(path.dirname(__file__), "segmentation.ui"
              ))


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setGeometry(0, 0, 1920, 1080)
        self.setWindowTitle("Segmentation")
        self.setupUi(self)
        init_ui(self)
        self.thresholding_method = "Otsu's"
        self.thresholding_type = "Local Thresholding"
        self.seeds = []

    def mouseDoubleClickEvent(self, event, img):
        if event.button() == Qt.LeftButton:
            img.read_image()

    def upload_image(self, img):
        img.read_image()

    def add_image_viewers(self, h_layout, images, images_labels):
        for i in range(len(images)):
            group_box = QGroupBox(images_labels[i])
            group_box_layout = QVBoxLayout(group_box)
            group_box_layout.addWidget(images[i])
            h_layout.addWidget(group_box)

    def apply_region_growing(self, image_input, image_output, seeds):
        threshold = self.rg_threshold_spinbox.value()
        result = image_input.apply_region_growing(
            image_input.img_original.copy(), threshold, seeds)
        image_output.display_image(result)

    def mousePressEvent(self, event, img):
        if event.button() == Qt.LeftButton:
            x = round((event.pos().x())/1.6)
            y = round((event.pos().y())/1.6)
            self.seeds.append((y, x))
            print(self.seeds)
            print(len(self.seeds))
            self.apply_region_growing(
                img, self.region_growing_img_output, self.seeds)
        elif event.button() == Qt.RightButton:
            self.seeds = []
            self.apply_region_growing(
                img, self.region_growing_img_output, self.seeds)

    def thresholding_tab(self, event, img):
        self.mouseDoubleClickEvent(event, img)

    def on_radio_button_clicked(self, button):
        if button.isChecked():
            self.thresholding_type = button.text()

    def on_combo_box_changed(self, index):
        selected_text = self.thresholding_types.currentText()
        self.thresholding_method = selected_text

    def apply_thersholding(self, image_input, image_output):
        thresholded_image = image_input.apply_thresholding(image_input.img_copy, self.thresholding_method, self.thresholding_type,
                                                           )
        image_output.display_image(thresholded_image)

    def apply_clustering(self, image_input, image_output):
        selected_clustering_type = self.clustering_types.currentText()
        parameter = self.clustering_spinBox.value()
        result = image_input.apply_clustering(
            image_input.img_original.copy(), selected_clustering_type, parameter)
        image_output.display_image(result)

    def exit_program(self):
        sys.exit()


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

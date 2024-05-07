from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from os import path
import sys
from Image import *
from Init_UI import *

FORM_CLASS, _ = loadUiType(
    path.join(path.dirname(__file__), "cvtask1.ui"))


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

    # noise tab
    def noise_slider_changed(self):
        slider_values = []
        slider_labels = [self.noise_label_1, self.noise_label_2]
        for slider in self.noise_sliders:
            slider_values.append(slider.value())
        noisy_image = self.noise_img.handle_slider_change(
            slider_values, slider_labels)
        self.noise_img.display_image(noisy_image)

    def noise_combo_change(self):
        selected_mode = self.noise_combobox.currentText()
        self.noise_img.handle_ui(
            selected_mode, self.noise_sliders, self.noise_labels)

    def filter_combo_change(self):
        selected_mode = self.filter_combobox.currentText()
        noise_image = self.noise_img.handle_ui_filter(
            selected_mode)
        self.noise_img.display_image(noise_image)

    # edge tab
    def edge_combo_changed(self):
        selected_edge = self.edge_combobox.currentText()
        if selected_edge=='Canny Edge Detection':
            img_1= self.edge_img_input.handle_edge_change(
                selected_edge)
            self.edge_img_output_1.display_image(img_1)
        else:
            img_1, img_2 = self.edge_img_input.handle_edge_change(
                selected_edge)
            self.edge_img_output_1.display_image(img_1)
            self.edge_img_output_2.display_image(img_2)

    # stats tab
    def stats_tab(self, event, img):
        self.mouseDoubleClickEvent(event, img)
        self.hist_img.file_path = self.stat_img_input.file_path
        self.dist_img.file_path = self.stat_img_input.file_path
        self.hist_img.plot_histogram()
        self.dist_img.plot_distribution_curve()

    def stats_mode_changed(self, mode):
        if mode == 'rgb':
            self.stat_img_input.display_image(
                self.stat_img_input.img_original)
        else:
            self.stat_img_input.display_image(self.stat_img_input.img_copy)

        self.stat_img_input.mode = mode
        self.hist_img.mode = mode
        self.dist_img.mode = mode
        self.hist_img.plot_histogram()
        self.dist_img.plot_distribution_curve()

    # equalize/normalize tab
    def equ_norm_tab(self, event, img):
        self.mouseDoubleClickEvent(event, img)
        normalized_img = self.eq_norm_img.normalize_image()
        equalized_img = self.eq_norm_img.equalize_image()
        self.normalized_img.display_image(normalized_img)
        self.equalized_img.display_image(equalized_img)

    # frequency domain tab
    def freq_tab(self, event, img, slider):
        self.mouseDoubleClickEvent(event, img)
        max_freq = img.get_max_freq()
        slider.setMinimum(0)
        slider.setMaximum(max_freq)
        slider.setSingleStep(int(max_freq/10))

    def freq_slider_changed(self):
        cutoff_freq = self.cutoff_slider.value()
        self.cutoff_label.setText(f"{cutoff_freq} Hz")
        low_pass_spectrum = self.freq_input.apply_filter(
            cutoff_freq, 'low_pass')
        high_pass_spectrum = self.freq_input.apply_filter(
            cutoff_freq, 'high_pass')
        low_pass_img = self.freq_input.inverse_fourier(
            low_pass_spectrum)
        high_pass_img = self.freq_input.inverse_fourier(
            high_pass_spectrum)
        self.freq_low.display_image(low_pass_img)
        self.freq_high.display_image(high_pass_img)

    # mixer tab
    def mix_slider_changed(self, slider, label):
        cutoff_freq = slider.value()
        label.setText(f"{cutoff_freq} Hz")
        cutoff_freq_1 = self.cutoff_slider_2.value()
        cutoff_freq_2 = self.cutoff_slider_3.value()
        if cutoff_freq_1 == 0 or cutoff_freq_2 == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Cut-off frequencies should be non zero values!")
            msg.setWindowTitle("Warning!")
            msg.exec_()
        else:
            lpf_restored = self.mix_input_1.apply_filter(
                cutoff_freq_1, 'low_pass')
            hpf_restored = self.mix_input_2.apply_filter(
                cutoff_freq_2, 'high_pass')
            res = self.mix_input_1.inverse_fourier(
                lpf_restored + hpf_restored)
            self.mix_output.display_image(res)

    # thresholding tab
    def thresh_tab(self, event, img):
        self.mouseDoubleClickEvent(event, img)
        thresh_global_img = self.thresh_input.manual_global_threshold()
        thresh_local_img = self.thresh_input.manual_local_threshold()
        self.thresh_global.display_image(thresh_global_img)
        self.thresh_local.display_image(thresh_local_img)

    def exit_program(self):
        sys.exit()


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

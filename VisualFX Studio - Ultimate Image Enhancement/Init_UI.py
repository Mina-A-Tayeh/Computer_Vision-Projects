from AddRemoveNoise import *
from EdgeDetector import *
from ImageEnhancer import *
from FrequancyDomain import *
from Thresholder import *
from ImageStats import *


def init_ui(self):
    noise_tab(self)
    edge_tab(self)
    stats_tab(self)
    equ_norm_tab(self)
    freq_tab(self)
    mixer_tab(self)
    threshold_tab(self)


def noise_tab(self):
    self.noise_sliders = [self.noise_slider_1, self.noise_slider_2]
    self.noise_labels = [self.noise_label_1, self.noise_label_2]
    self.noise_img = AddRemoveNoise()
    self.add_image_viewers(self.horizontalLayout_9,
                           [self.noise_img], ['Image viewer'])
    self.noise_slider_1.valueChanged.connect(self.noise_slider_changed)
    self.noise_slider_2.valueChanged.connect(self.noise_slider_changed)
    self.filter_combobox.currentIndexChanged.connect(
        self.filter_combo_change)
    self.noise_combobox.currentIndexChanged.connect(
        self.noise_combo_change)
    self.noise_img.mouseDoubleClickEvent = lambda event: self.mouseDoubleClickEvent(
        event, self.noise_img)
    self.reset_button.clicked.connect(self.noise_img.reset_changes)


def edge_tab(self):
    self.edge_img_input = EdgeDetector()
    self.edge_img_output_1 = EdgeDetector()
    self.edge_img_output_2 = EdgeDetector()
    edge_images = [self.edge_img_input,
                   self.edge_img_output_1, self.edge_img_output_2]
    edge_labels = ['Image viewer',
                   'Edges in X-direction', 'Edges in Y-direction']
    self.add_image_viewers(self.horizontalLayout_4,
                           edge_images, edge_labels)
    self.edge_img_input.mouseDoubleClickEvent = lambda event: self.mouseDoubleClickEvent(
        event, self.edge_img_input)
    self.edge_combobox.currentIndexChanged.connect(self.edge_combo_changed)


def stats_tab(self):
    self.stat_img_input = ImageStats()
    self.hist_img = ImageStats()
    self.dist_img = ImageStats()
    stat_images = [self.stat_img_input,
                   self.hist_img, self.dist_img]
    stat_labels = ['Input viewer',
                   'Histogram', 'Distribution curve']
    self.add_image_viewers(self.horizontalLayout_14,
                           stat_images, stat_labels)
    self.stat_img_input.mouseDoubleClickEvent = lambda event: self.stats_tab(
        event, self.stat_img_input)
    self.grey_button.clicked.connect(lambda: self.stats_mode_changed('gray'))
    self.rgb_button.clicked.connect(lambda: self.stats_mode_changed('rgb'))


def equ_norm_tab(self):
    self.eq_norm_img = ImageEnhancer()
    self.equalized_img = ImageEnhancer()
    self.normalized_img = ImageEnhancer()
    eq_norm_images = [self.eq_norm_img,
                      self.equalized_img, self.normalized_img]
    eq_norm_labels = ['Image viewer',
                      'Equalized image', 'Normalized image']
    self.add_image_viewers(self.horizontalLayout_13,
                           eq_norm_images, eq_norm_labels)
    self.eq_norm_img.mouseDoubleClickEvent = lambda event: self.equ_norm_tab(
        event, self.eq_norm_img)


def freq_tab(self):
    self.freq_input = FrequencyDomain()
    self.freq_high = FrequencyDomain()
    self.freq_low = FrequencyDomain()
    freq_images = [self.freq_input,
                   self.freq_high, self.freq_low]
    freq_labels = ['Input viewer',
                   'High pass filter', 'Low pass filter']
    self.add_image_viewers(self.horizontalLayout_15,
                           freq_images, freq_labels)
    self.freq_input.mouseDoubleClickEvent = lambda event: self.freq_tab(
        event, self.freq_input, self.cutoff_slider)
    self.cutoff_slider.valueChanged.connect(self.freq_slider_changed)


def mixer_tab(self):
    self.mix_input_1 = FrequencyDomain()
    self.mix_input_2 = FrequencyDomain()
    self.mix_output = FrequencyDomain()
    mix_images = [self.mix_input_1,
                  self.mix_input_2, self.mix_output]
    mix_labels = ['Input viewer 1',
                  'Input viewer 2', 'Output viewer']
    self.add_image_viewers(self.horizontalLayout_17,
                           mix_images, mix_labels)
    self.mix_input_1.mouseDoubleClickEvent = lambda event: self.freq_tab(
        event, self.mix_input_1, self.cutoff_slider_2)
    self.mix_input_2.mouseDoubleClickEvent = lambda event: self.freq_tab(
        event, self.mix_input_2, self.cutoff_slider_3)
    self.cutoff_slider_2.valueChanged.connect(
        lambda: self.mix_slider_changed(self.cutoff_slider_2, self.cutoff_label_2))
    self.cutoff_slider_3.valueChanged.connect(
        lambda: self.mix_slider_changed(self.cutoff_slider_3, self.cutoff_label_3))


def threshold_tab(self):
    self.thresh_input = Thresholder()
    self.thresh_local = Thresholder()
    self.thresh_global = Thresholder()
    thresh_images = [self.thresh_input,
                     self.thresh_local, self.thresh_global]
    thresh_labels = ['Image viewer',
                     'Local thresholding', 'Global thresholding']
    self.add_image_viewers(self.horizontalLayout_19,
                           thresh_images, thresh_labels)
    self.thresh_input.mouseDoubleClickEvent = lambda event: self.thresh_tab(
        event, self.thresh_input)

import numpy as np


def fit_polynomial(binary_image, left_x, left_y, right_x, right_y):
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    plot_y = np.linspace(0, binary_image.shape[0] - 1, binary_image.shape[0])
    left_fit_x = left_fit[0] * (plot_y ** 2) + left_fit[1] * plot_y + left_fit[2]
    right_fit_x = right_fit[0] * (plot_y ** 2) + right_fit[1] * plot_y + right_fit[2]
    return plot_y, left_fit, left_fit_x, right_fit, right_fit_x


def single_fit(binary_image, sing_x, sing_y):
    left_fit = np.polyfit(sing_y, sing_x, 2)
    plot_y = np.linspace(0, binary_image.shape[0] - 1, binary_image.shape[0])
    left_fit_x = left_fit[0] * (plot_y ** 2) + left_fit[1] * plot_y + left_fit[2]
    return plot_y, left_fit, left_fit_x

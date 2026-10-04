import numpy as np


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def d_mse(y_pred, y_true):
    return 2 * (y_pred - y_true) / np.size(y_true)
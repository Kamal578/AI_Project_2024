import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def max_dif(y):
    return np.max(y) - np.min(y)

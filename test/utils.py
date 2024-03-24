import numpy as np


def mae(y_true, y_pred):
    """ Mean Absolute Error """
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    """ Mean Squared Error """
    return np.mean((y_true - y_pred) ** 2)


def max_dif(y):
    """ Maximum difference between the target values """
    return np.max(y) - np.min(y)


def categorical_accuracy(y_true, y_pred):
    """ Categorical accuracy """
    return np.mean(y_true == y_pred)


def mae_accuracy(m, y_true, y_pred):
    """
    Mean Absolute Error based Accuracy

    :param m: maximum difference between the target values
    :param y_true: true target values
    :param y_pred: predicted target values
    :return: accuracy
    """
    return 1 - mae(y_true, y_pred) / m


def mse_accuracy(m, y_true, y_pred):
    """
    Mean Squared Error based Accuracy

    :param m: maximum difference between the target values
    :param y_true: true target values
    :param y_pred: predicted target values
    :return: accuracy
    """
    return 1 - mse(y_true, y_pred) / m ** 2

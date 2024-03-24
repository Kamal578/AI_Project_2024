import numpy as np


def mae(y_true, y_pred):
    """ Mean Absolute Error (for regression) """
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    """ Mean Squared Error (for regression) """
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """ Root Mean Squared Error (for regression) """
    return np.sqrt(mse(y_true, y_pred))


def accuracy(y_true, y_pred):
    """ Accuracy (for multiclass classification) """
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, average='macro'):
    """Precision (for multiclass classification)

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    average (str, optional): Type of averaging to perform.
        - 'macro': Calculate precision for each class and return the unweighted mean.
        - 'micro': Calculate precision globally by counting the total true positives, false positives, and false negatives.

    Returns:
    float or array-like: Precision score(s).
    """
    # Convert inputs to numpy arrays if they are not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle binary classification
    if len(np.unique(y_true)) == 2:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Handle multiclass classification
    elif average == 'macro':
        # Calculate precision for each class and return the unweighted mean
        unique_classes = np.unique(y_true)
        precisions = []
        for cls in unique_classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions.append(precision_cls)
        return np.mean(precisions)
    elif average == 'micro':
        # Calculate precision globally by counting the total true positives, false positives, and false negatives
        tp = np.sum((y_true == y_pred) & (y_pred == 1))
        fp = np.sum((y_true != y_pred) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    else:
        raise ValueError("Invalid value for 'average'. It must be 'macro' or 'micro'.")


def recall(y_true, y_pred, average='macro'):
    """Recall (for multiclass classification)

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    average (str, optional): Type of averaging to perform.
        - 'macro': Calculate recall for each class and return the unweighted mean.
        - 'micro': Calculate recall globally by counting the total true positives, false positives, and false negatives.
        - None: Calculate recall for each class separately.

    Returns:
    float or array-like: Recall score(s).
    """

    # Convert inputs to numpy arrays if they are not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle binary classification
    if len(np.unique(y_true)) == 2:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Handle multiclass classification
    if average == 'macro':
        # Calculate recall for each class and return the unweighted mean
        unique_classes = np.unique(y_true)
        recalls = []
        for cls in unique_classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            recall_cls = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall_cls)
        return np.mean(recalls)
    elif average == 'micro':
        # Calculate recall globally by counting the total true positives, false positives, and false negatives
        tp = np.sum((y_true == y_pred) & (y_pred == 1))
        fn = np.sum((y_true != y_pred) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        raise ValueError("Invalid value for 'average'. It must be 'macro' or 'micro'.")


def f1_score(y_true, y_pred, average='macro'):
    """F1 Score (for multiclass classification)

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    average (str, optional): Type of averaging to perform.
        - 'macro': Calculate F1 score for each class and return the unweighted mean.
        - 'micro': Calculate F1 score globally by counting the total true positives, false positives, and false negatives.

    Returns:
    float or array-like: F1 score(s).
    """
    # Calculate precision and recall
    precision_val = precision(y_true, y_pred, average)
    recall_val = recall(y_true, y_pred, average)

    # Handle division by zero
    if precision_val == 0 or recall_val == 0:
        return 0.0

    # Calculate F1 score
    f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val)
    return f1

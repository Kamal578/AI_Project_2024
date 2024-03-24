from abc import ABC
import numpy as np

from src.BaseModel import BaseModel


class BaseDecisionTree(BaseModel, ABC):
    """ Base Decision Tree """

    def __init__(self, max_depth, min_samples_split, name=None):
        """
        Initialize the DecisionTree object.

        :param max_depth: Maximum depth of the tree. If None, the tree grows until all leaves are pure.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        """
        super().__init__()
        self.name = name

        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.n_features = None

    def fit(self, X, y):
        """
        Fit the model to the training data.

        :param X: Training data, a numpy array where each row represents a sample and each column represents a feature.
        :param y: Target values, a numpy array containing the labels for each sample in X.
        """
        self.n_features = X.shape[1]

        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        """
        Predict the target values for input data.

        :param X: Input data, a numpy array where each row represents a sample and each column represents a feature.
        :return: Predicted target values for each sample in X.
        """
        return np.array([self._predict(inputs) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree.

        :param X: Input data for the current node.
        :param y: Target values for the current node.
        :param depth: Current depth of the tree.
        :return: Node of the tree if it is a leaf, otherwise a tuple of the split feature, threshold, left and right trees.
        """
        n_samples = X.shape[0]
        mean = np.mean(y)

        # Check termination conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (self.min_samples_split is not None and n_samples < self.min_samples_split):
            return mean

        # Find the best feature and threshold for splitting
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return mean

        # Split the data based on the best feature and threshold
        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices

        # Recursively grow the left and right subtrees
        left_tree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return feature, threshold, left_tree, right_tree

    def _predict(self, inputs):
        """
        Recursively predict the target value

        :param inputs: input data
        :return: predicted target value
        """
        node = self.tree
        while not isinstance(node, np.float64):
            feature, threshold, left_tree, right_tree = node
            if inputs[feature] < threshold:
                node = left_tree
            else:
                node = right_tree
        return node

    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split the data

        :param X: input data
        :param y: target values
        :return: best feature and threshold
        """
        m, n = X.shape
        if m <= 1:
            return None, None

        # calculate the variance of the target values
        variance = np.var(y)
        best_variance = variance
        best_feature = None
        best_threshold = None

        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                left_variance = np.var(y[left_indices])
                right_variance = np.var(y[right_indices])
                current_variance = left_variance + right_variance

                if current_variance < best_variance:
                    best_variance = current_variance
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold


class DecisionTreeRegressor(BaseDecisionTree):
    """ Decision Tree Regressor """

    def __init__(self, max_depth=None, min_samples_split=2, name=None):
        super().__init__(max_depth, min_samples_split, name=name)

    def fit(self, X, y):
        """
        Fit the model to the training data

        :param X: training data
        :param y: target values
        """
        y = y.astype(np.float64)
        super().fit(X, y)

    def predict(self, X):
        """
        Predict the target values

        :param X: input data
        :return: predicted target values
        """
        return super().predict(X).astype(np.float64)


class DecisionTreeClassifier(BaseDecisionTree):
    """ Decision Tree Classifier """

    def __init__(self, max_depth=None, min_samples_split=2, name=None):
        super().__init__(max_depth, min_samples_split, name=name)

    def fit(self, X, y):
        """
        Fit the model to the training data

        :param X: training data
        :param y: target values
        """
        y = y.astype(np.int64)
        super().fit(X, y)

    def predict(self, X):
        """
        Predict the target values

        :param X: input data
        :return: predicted target values
        """
        return super().predict(X).astype(np.int64)

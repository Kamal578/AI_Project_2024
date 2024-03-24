import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=None):
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.n_classes = None
        self.n_features = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]

        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        n_labels = [np.sum(y == i) for i in range(self.n_classes)]
        most_common_label = np.argmax(n_labels)

        if (self.max_depth is not None and depth >= self.max_depth) or \
           (self.min_samples_split is not None and n_samples < self.min_samples_split) or \
           np.all(y == y[0]):
            return most_common_label

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return most_common_label

        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices

        left_tree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return feature, threshold, left_tree, right_tree

    def _best_split(self, X, y):
        best_gini = 1
        best_feature = None
        best_threshold = None

        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices

                gini = self._gini(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini(self, y_left, y_right):
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        gini_left = 1 - sum([(np.sum(y_left == i) / n_left) ** 2 for i in range(self.n_classes)])
        gini_right = 1 - sum([(np.sum(y_right == i) / n_right) ** 2 for i in range(self.n_classes)])

        return (n_left / n_total) * gini_left + (n_right / n_total) * gini_right

    def _predict(self, inputs):
        node = self.tree
        while isinstance(node, tuple):
            feature, threshold, left_tree, right_tree = node
            if inputs[feature] < threshold:
                node = left_tree
            else:
                node = right_tree

        return node

    def __repr__(self):
        return str(self.tree)

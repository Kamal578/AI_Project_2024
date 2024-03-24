import numpy as np

class DecisionTree:
    """ 
    Decision Tree Classifier 
    
    This class implements a simple decision tree classifier for binary classification tasks.
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Initialize the DecisionTree object.

        :param max_depth: Maximum depth of the tree. If None, the tree grows until all leaves are pure.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        """
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.n_classes = None
        self.n_features = None

    def fit(self, X, y):
        """
        Fit the model to the training data.

        :param X: Training data, a numpy array where each row represents a sample and each column represents a feature.
        :param y: Target values, a numpy array containing the labels for each sample in X.
        """
        self.n_classes = len(np.unique(y))
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
        n_labels = [np.sum(y == i) for i in range(self.n_classes)]
        most_common_label = np.argmax(n_labels)

        # Check termination conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (self.min_samples_split is not None and n_samples < self.min_samples_split) or \
           np.all(y == y[0]):
            return most_common_label

        # Find the best feature and threshold for splitting
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return most_common_label

        # Split the data based on the best feature and threshold
        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices

        # Recursively grow the left and right subtrees
        left_tree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return feature, threshold, left_tree, right_tree

    def _best_split(self, X, y):
        """
        Find the best feature and threshold for splitting the data.

        :param X: Input data.
        :param y: Target values.
        :return: The best feature and threshold to split the data.
        """
        best_gini = 1
        best_feature = None
        best_threshold = None

        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices

                # Calculate Gini impurity for the split
                gini = self._gini(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini(self, y_left, y_right):
        """
        Calculate the Gini impurity for a split.

        :param y_left: Target values of the left node.
        :param y_right: Target values of the right node.
        :return: Gini impurity for the split.
        """
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        # Check if either n_left or n_right is zero to avoid division by zero
        if n_left == 0 or n_right == 0:
            return 0  # If either node has no samples, return 0 impurity

        # Calculate Gini impurity
        gini_left = 1 - sum([(np.sum(y_left == i) / n_left) ** 2 for i in range(self.n_classes)])
        gini_right = 1 - sum([(np.sum(y_right == i) / n_right) ** 2 for i in range(self.n_classes)])

        return (n_left / n_total) * gini_left + (n_right / n_total) * gini_right

    def _predict(self, inputs):
        """
        Predict the target value of the input data.

        :param inputs: Input data.
        :return: Predicted target value.
        """
        node = self.tree
        while isinstance(node, tuple):
            feature, threshold, left_tree, right_tree = node
            if inputs[feature] < threshold:
                node = left_tree
            else:
                node = right_tree

        return node

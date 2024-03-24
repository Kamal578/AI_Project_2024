import numpy as np

from src.DecisionTree import DecisionTree


class RandomForest:
    """ Random Forest Classifier """

    def __init__(self, n_trees, max_depth, min_samples_split, data_per_tre=0.8):
        """
        :param n_trees: number of trees in the forest
        :param max_depth: maximum depth of the tree
        :param min_samples_split: minimum number of samples required to split an internal node
        :param data_per_tre: fraction of the training data to be used for each tree
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.data_per_tree = data_per_tre

        self.trees = [DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split) for _ in range(n_trees)]

    def fit(self, X, y):
        """
        Fit the model to the training data

        :param X: training data
        :param y: target values
        """
        # fit each tree to a random subset of the training data
        for tree in self.trees:
            indices = np.random.choice(X.shape[0], size=int(self.data_per_tree * X.shape[0]), replace=True)
            tree.fit(X[indices], y[indices])

    def predict(self, X):
        """
        Predict the target values

        :param X: input data
        :return: predicted target values
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.argmax(np.bincount(predictions[:, i])) for i in range(X.shape[0])])

import numpy as np

from src.DecisionTree import DecisionTree


class RandomForest:
    """ Random Forest Classifier """

    def __init__(self, n_trees=100, max_depth=None, min_samples_split=None):
        """
        :param n_trees: number of trees in the forest
        :param max_depth: maximum depth of the tree
        :param min_samples_split: minimum number of samples required to split an internal node
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.trees = [DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split) for _ in range(n_trees)]

    def fit(self, X, y):
        """
        Fit the model to the training data

        :param X: training data
        :param y: target values
        """
        for tree in self.trees:
            tree.fit(X, y)

    def predict(self, X):
        """
        Predict the target values

        :param X: input data
        :return: predicted target values
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.argmax(np.bincount(predictions[:, i])) for i in range(X.shape[0])])

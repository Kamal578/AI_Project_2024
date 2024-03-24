import numpy as np
from src.DecisionTree import DecisionTree  # Assuming DecisionTree class is imported from a file named DecisionTree.py


class RandomForest:
    """ 
    Random Forest Classifier 
    
    This class implements a random forest classifier, which is an ensemble learning method 
    that operates by constructing a multitude of decision trees during training and outputs 
    the class that is the mode of the classes (classification) or mean prediction (regression) 
    of the individual trees.
    """

    def __init__(self, n_trees, max_depth, min_samples_split, data_per_tree=0.8):
        """
        Initialize the RandomForest object.

        :param n_trees: Number of trees in the forest.
        :param max_depth: Maximum depth of each tree.
        :param min_samples_split: Minimum number of samples required to split an internal node in each tree.
        :param data_per_tree: Fraction of the training data to be used for each tree.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.data_per_tree = data_per_tree

        # Create a list to hold the decision trees
        self.trees = [DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split) for _ in range(n_trees)]

    def fit(self, X, y):
        """
        Fit the model to the training data.

        :param X: Training data, a numpy array where each row represents a sample and each column represents a feature.
        :param y: Target values, a numpy array containing the labels for each sample in X.
        """
        # Fit each tree to a random subset of the training data
        for tree in self.trees:
            indices = np.random.choice(X.shape[0], size=int(self.data_per_tree * X.shape[0]), replace=True)
            tree.fit(X[indices], y[indices])

    def predict(self, X):
        """
        Predict the target values for input data.

        :param X: Input data, a numpy array where each row represents a sample and each column represents a feature.
        :return: Predicted target values for each sample in X.
        """
        # Make predictions with each tree in the forest
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Aggregate predictions by taking the mode (most common) of the predictions of all trees
        return np.array([np.argmax(np.bincount(predictions[:, i])) for i in range(X.shape[0])])

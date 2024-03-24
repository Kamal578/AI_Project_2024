from abc import ABC
import numpy as np

from src.BaseModel import BaseModel

from src.DecisionTree import DecisionTreeRegressor, DecisionTreeClassifier


class BaseRandomForest(BaseModel, ABC):
    """ Base class for Random Forest Regressor and Classifier """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None, name=None):
        super().__init__()
        self.name = name

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        """
        Fit the model to the training data

        :param X: training data
        :param y: target values
        """
        n_samples, n_features = X.shape

        for _ in range(self.n_estimators):
            tree = self._get_tree()
            # Randomly select rows and features for each tree
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            feature_indices = self._get_random_features(n_features)
            tree.fit(X[sample_indices][:, feature_indices], y[sample_indices])
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the target values

        :param X: input data
        :return: predicted target values
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return self._aggregate_predictions(predictions)

    def _get_tree(self):
        raise NotImplementedError

    def _aggregate_predictions(self, predictions):
        raise NotImplementedError

    def _get_random_features(self, n_features):
        indices = np.arange(n_features)
        if self.max_features is None:
            return indices

        indices = np.random.choice(indices, self.max_features, replace=False)
        return indices


class RandomForestRegressor(BaseRandomForest):
    """ Random Forest Regressor """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None, name=None):
        super().__init__(n_estimators, max_depth, min_samples_split, max_features, name=name)

    def _get_tree(self):
        return DecisionTreeRegressor(self.max_depth, self.min_samples_split)

    def _aggregate_predictions(self, predictions):
        return np.mean(predictions, axis=0)


class RandomForestClassifier(BaseRandomForest):
    """ Random Forest Classifier """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None, name=None):
        super().__init__(n_estimators, max_depth, min_samples_split, max_features, name=name)

    def _get_tree(self):
        return DecisionTreeClassifier(self.max_depth, self.min_samples_split)

    def _aggregate_predictions(self, predictions):
        return np.round(np.mean(predictions, axis=0))

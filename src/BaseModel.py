from abc import ABC


class BaseModel(ABC):
    """ Base class for all models """

    def __init__(self):
        self.name = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def __str__(self):
        if self.name is not None:
            return self.name
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

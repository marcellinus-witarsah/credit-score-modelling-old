from abc import ABC, abstractmethod


class ModelStrategy(ABC):
    # If the model is not implemented an error will raised
    @property
    def model(self):
        raise NotImplementedError

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_true):
        pass

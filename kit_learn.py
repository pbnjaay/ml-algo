from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    def __init__(self) -> None:
        self.mu = None
        self.sigma = None

    def _zscore_normalize(self, X) -> None:
        mu = np.mean(X, axis=0)                 
        sigma = np.std(X, axis=0, ddof=1)         
        X_norm = (X - mu) / sigma     

        return X_norm, mu, sigma
    
    @abstractmethod
    def fit(self, X, y) ->None:
        pass

    @abstractmethod     
    def predict(self, X):
        pass


class LinearRegression(BaseModel):
    def __init__(self, alpha = 0.001, iter=1000, lambda_=0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.lambda_ = lambda_
        self.iter = iter
        self.w = None
        self.b = None

    def fit(self, X, y) -> None:
        X, self.mu, self.sigma = self._zscore_normalize(X)
        m = X.shape[0]
        n = X.shape[1]
        self.w = np.zeros(n)
        self.b = 0

        for i in range(self.iter):
            y_pred = np.dot(X, self.w) + self.b
            residue = y_pred - y
            self.w -= self.alpha * (1/m) * np.sum(np.dot(X.T, residue)) + (self.lambda_/m) * self.w
            self.b -= self.alpha * (1/m)  * np.sum(residue)

    def predict(self, x):
        x_norm = (x - self.mu) / self.sigma

        return np.dot(self.w, x_norm.T) + self.b
    
    def r2_score(self, y_pred, y):
        SS_res = np.sum(np.square(y - y_pred))
        SS_tot = np.sum(np.square(y - np.mean(y)))

        return 1 - (SS_res / SS_tot)

class LogisticRegression(BaseModel):
    def __init__(self, alpha=0.01, iter=1000, lambda_=0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.lambda_ = lambda_
        self.iter = iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        X, self.mu, self.sigma = self._zscore_normalize(X)
        m = X.shape[0]
        n = X.shape[1]
        self.w = np.zeros(n)
        self.b = 0

        for i in range(self.iter):
            z = np.dot(X, self.w) + self.b
            y_pred = self.__sigmoid(z)
            residue = y_pred - y
            self.w -= self.alpha * (1/m) * np.sum(np.dot(X.T, residue)) + (self.lambda_/m) * self.w
            self.b -= self.alpha * (1/m)  * np.sum(residue)


    def predict(self, x):
        x_norm = (x - self.mu) / self.sigma
        z = np.dot(self.w, x_norm.T) + self.b
        y_pred = self.__sigmoid(z) 
        return [1 if y_pred[i] > 0.5 else 0 for i in range(y_pred.shape[0])]

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
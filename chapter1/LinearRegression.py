import numpy as np


class LinearRegression:
    def __init__(self , lr = 0.01, iterations = 500):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self , X ,y):
        self.bias = 0
        self.weights = np.zeros(X.shape[1])

        for _ in range(self.iterations):
            pred = self.predict(X)
            error =  y - pred

            dw = (1 / X.shape[0]) * (X.T @ error)       # для weights
            db = (1 / X.shape[0]) * np.sum(error)        # для bias

            self.weights += self.lr * dw
            self.bias += self.lr * db



    def predict(self,X):
        prediction = X @ self.weights + self.bias
        return prediction


    def mse(self,X,y):
        return np.mean((y - self.predict(X)) ** 2 )






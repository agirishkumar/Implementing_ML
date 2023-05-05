import numpy as np


class UnivariateLinearRegression:
    def __init__(self, learning_rate=0.0001):
        self.lr = learning_rate
        self.theta_0 = None
        self.theta_1 = None

    def train(self, X, y):

        self.theta_1 = 0
        self.theta_0 = 0

        convergence_threshold = 1e-4
        change = np.inf
        prev_mse = np.inf
        max_iter = 10000
        iter_count = 0

        while change > convergence_threshold and iter_count < max_iter:
            y_pred = X * self.theta_1 + self.theta_0
            d_theta_1 = np.mean((y_pred - y) * X)
            d_theta_0 = np.mean(y_pred - y)

            prev_theta_0 = self.theta_0
            prev_theta_1 = self.theta_1

            self.theta_0 = self.theta_0 - (self.lr * d_theta_0)
            self.theta_1 = self.theta_1 - (self.lr * d_theta_1)

            change = np.sqrt((self.theta_0 - prev_theta_0) ** 2 + (self.theta_1 - prev_theta_1) ** 2)

            # Early stopping based on mean squared error
            mse = np.mean((y_pred - y) ** 2)
            if mse > prev_mse:
                break

            prev_mse = mse
            iter_count += 1

    def predict(self, X):
        y_pred = X * self.theta_1 + self.theta_0
        return y_pred

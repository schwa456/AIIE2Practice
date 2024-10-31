from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np
import pandas as pd
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper

class LOOCV:
    def __init__(self, model):
        """
        LOOCVS(Leave-One-Out Cross Validation) Class
        :param model: model that is used to learning
        """
        self.model = model
        self.mse_errors = []
        self.mape_errors = []
        self.r2_error = None

    @timer
    def fit(self, X, y):
        """
        execute LOOCV for df
        :param X: X data as nd array
        :param y: name of target column
        :return: score measured by scoring metrics
        """

        n_samples, n_features = X.shape

        for i in range(n_samples):
            mask = np.arange(n_samples) != i
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[mask].values
                X_test = X.iloc[[i]].values
            else:
                X_train = X[mask]
                X_test = X[i].reshape(1, 2)

            if isinstance(y, pd.Series):
                y_train = y.iloc[mask].values
                y_test = y.iloc[i]
            else:
                y_train = y[mask]
                y_test = y[i]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            mse_error = self._calculate_score([y_test], y_pred, 'mse')
            self.mse_errors.append(mse_error)
            mape_error = self._calculate_score([y_test], y_pred, 'mape')
            self.mape_errors.append(mape_error)
            self.r2_error = r2_score([y_test], y_pred)

    def _calculate_score(self, y_test, y_pred, scoring):
        """
        calculate score in the designated way
        :param y_test: y_test data
        :param y_pred: y_prediction data
        :return: score
        """
        if scoring == 'mape':
            return mean_absolute_percentage_error(y_test, y_pred)
        elif scoring == 'mse':
            return mean_squared_error(y_test, y_pred)
        elif scoring == 'r2':
            r2_error = r2_score(y_test, y_pred)
            return r2_error
        else:
            raise ValueError("Not supported Metrics. Use 'mse' or 'mape'.")

    def mean_score(self, scoring):
        """
        calculate mean score
        :param scoring: scoring metrics
        :return: mean_score
        """

        if scoring == 'mape':
            return np.mean(self.mape_errors)
        elif scoring == 'mse':
            return np.mean(self.mse_errors)
        elif scoring == 'r2':
            return self.r2_error

def __main__():
    # 데이터 생성
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 4, 9, 16, 25])

    # LOOCV 클래스 사용
    model = LinearRegression()
    loocv = LOOCV(model)  # MSE를 성능 지표로 사용
    loocv.fit(X, y)
    print(f"Mean Squared Error: {loocv.mean_score('mse'):.4f}")
    print(f"R Squared Error: {loocv.mean_score('r2'):.4f}")
    print(f"Mean Absolute Percentage Error: {loocv.mean_score('mape'):.4f}")

if __name__ == '__main__':
    __main__()
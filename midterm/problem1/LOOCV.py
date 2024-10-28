from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.base import clone
import numpy as np

class LOOCV:
    def __init__(self, model, scoring='mse'):
        """
        LOOCVS(Leave-One-Out Cross Validation) Class
        :param model: model that is used to learning
        :param scoring: scoring metrics ('mse', 'r2', 'mape')
        """
        self.model = model
        self.scoring = scoring
        self.errors = []

    def split(self, X):
        """
        Split the Data in the LOO way
        :param X: Data
        :return: None
        """
        n_samples = len(X)
        for i in range(n_samples):
            train_index = np.delete(np.arange(n_samples), i)
            test_index = np.array([i])
            yield train_index, test_index

    def fit_predict(self, X, y):
        """
        Learns model and predict
        :param X: input data
        :param y: target data
        :return: None
        """

        for train_index, test_index in self.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model_clone = clone(self.model)
            model_clone.fit(X_train, y_train)

            y_pred = model_clone.predict(X_test)
            error = self._calculate_score(y_test, y_pred)
            self.errors.append(error)

    def _calculate_score(self, y_test, y_pred):
        """
        calculate score in the designated way
        :param y_test: y_test data
        :param y_pred: y_prediction data
        :return: score
        """
        if self.scoring == 'r2':
            return r2_score(y_test, y_pred)
        elif self.scoring == 'mape':
            return mean_absolute_percentage_error(y_test, y_pred)
        elif self.scoring == 'mse':
            return mean_squared_error(y_test, y_pred)
        else:
            raise ValueError("Not supported Metrics. Use 'mse', 'mape', or 'r2'.")

    def mean_score(self):
        """
        calculate mean score
        :return: mean_score
        """
        return np.mean(self.errors)

def __main__():
    # 데이터 생성
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 4, 9, 16, 25])

    # LOOCV 클래스 사용
    model_mse = LinearRegression()
    loocv_mse = LOOCV(model_mse, scoring='mse')  # MSE를 성능 지표로 사용
    loocv_mse.fit_predict(X, y)
    print(f"Mean Squared Error: {loocv_mse.mean_score():.4f}")

    model_mape = LinearRegression()
    loocv_mape = LOOCV(model_mape, scoring='mape')  # MAPE를 성능 지표로 사용
    loocv_mape.fit_predict(X, y)
    print(f"Mean Absolute Percentage Error: {loocv_mape.mean_score():.4f}")

    model_r2 = LinearRegression()
    loocv_r2 = LOOCV(model_r2, scoring='r2')  # R2를 성능 지표로 사용
    loocv_r2.fit_predict(X, y)
    print(f"R Squared Error: {loocv_r2.mean_score():.4f}")

if __name__ == '__main__':
    __main__()
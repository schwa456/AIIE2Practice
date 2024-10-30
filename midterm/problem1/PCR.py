import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

class PCR:
    def __init__(self):
        self.model = LinearRegression()
        self.coefficients_ = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self.coefficients_ = self.model.coef_

    def predict(self, X):
        return self.model.predict(X)

    def get_coefficients(self):
        if self.coefficients_ is not None:
            return self.coefficients_
        else:
            raise ValueError("PCR is not fitted yet, Use fit() method first.")

    def score(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, mape, r2

def __main__():
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = X @ np.array([1.5, -2, 0.5, 0, 1, 0, 0, 0, 0, 2]) + np.random.randn(100)

    # PCR 클래스 사용
    pcr = PCR()  # 목표 누적 분산 비율 95% 설정
    pcr.fit(X, y)  # 모델 학습

    # 테스트 데이터 생성 및 예측
    X_test = np.random.rand(20, 10)
    y_test = X_test @ np.array([1.5, -2, 0.5, 0, 1, 0, 0, 0, 0, 2]) + np.random.randn(20)

    # 모델 성능 평가
    mse, mape, r2 = pcr.score(X_test, y_test)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.4f}")
    print(f"R² Score: {r2:.4f}")

if __name__ == '__main__':
    __main__()
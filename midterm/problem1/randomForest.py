import numpy as np
import pandas as pd
from collections import Counter
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error


class RandomForestRegressor:
    def __init__(self, max_feature=None, n_estimators=10, max_depth=None, random_state=None):
        """
        Random Forest Regressor
        :param n_estimators: Number of Tree
        :param max_feature: Number of feature of each tree
        :param max_depth: max depth of each tree
        :param random_state: seed for random number generator
        """

        self.n_estimators = n_estimators
        self.max_feature = max_feature
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = [] # list for Trees and features

    def _sample_data(self, X, y):
        """
        randomly sample data using bootstrap sampling
        :param X: X data
        :param y: y data
        :return: randomly sampled data
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)

        X_sample = X[indices]
        y_sample = y[indices]

        if X_sample.size == 0 or y_sample.size == 0:
            raise ValueError("Sampled data is empty")

        return X_sample, y_sample

    def _select_feature(self, X):
        """
        select feature for each tree randomly
        :param X: X data
        :return: selected feature
        """

        n_features = X.shape[1]
        num_features = self.max_feature or max(1, n_features // 3)
        selected_features = np.random.choice(n_features, num_features, replace=False)

        if len(selected_features) == 0:
            raise ValueError("No features selected")

        return selected_features

    def fit(self, X, y):
        """
        fitting Random Forest model
        """
        np.random.seed(self.random_state)

        for _ in range(self.n_estimators):
            # selecting bootstrap sampling and feature
            X_sample, y_sample = self._sample_data(X, y)
            selected_features = self._select_feature(X_sample)
            X_sample = X_sample[:, selected_features]

            if X_sample.size == 0 or y_sample.size == 0:
                raise ValueError("샘플링된 데이터가 비어 있습니다.")

            # fitting Tree
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_sample, y_sample)

            self.trees.append((tree, selected_features))

    def predict(self, X):
        """
        make a prediction
        """
        predictions = np.zeros((X.shape[0], len(self.trees)))

        for i, (tree, features) in enumerate(self.trees):
            if isinstance(X, pd.DataFrame):
                X_subset = X.iloc[:, features].values
            else:
                X_subset = X[:, features]
            predictions[:, i] = tree.predict(X_subset)

        return np.mean(predictions, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, mape, r2

def __main__():
    X = np.random.rand(100, 2)
    y = np.random.rand(100)

    rf_regressor = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    rf_regressor.fit(X, y)

    y_pred = rf_regressor.predict(X)
    print(f"Predicted Target Value : {y_pred}")


    mse, mape, r2 = rf_regressor.score(X, y)
    print(f"MSE : {mse}")
    print(f"MAPE : {mape}")
    print(f"R2 Score : {r2}")

if __name__ == '__main__':
    __main__()
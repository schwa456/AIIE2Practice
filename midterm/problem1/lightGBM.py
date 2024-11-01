import joblib
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import time
from sklearn.model_selection import GridSearchCV


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper

class LightGBMRegressor:
    def __init__(self, params=None, ):
        params = {
            'objective': 'regression',
            'boosting': 'gbdt',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'verbose': -1,
        }

        self.params = params
        self.model = None

    @timer
    def fit(self, X_train, y_train, X_test=None, y_test=None, num_boost_round = 1000):
        train_data = lgb.Dataset(X_train, y_train)
        valid_sets = [train_data]

        if X_test is not None and y_test is not None:
            valid_data = lgb.Dataset(X_test, y_test)
            valid_sets.append(valid_data)

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
        )

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before predicting.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def score(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, mape, r2

    def save_model(self, file_name):
        if self.model is None:
            raise ValueError("Model must be fitted before saving.")
        joblib.dump(self.model, file_name)
        print(f"Model saved at {file_name}")

    def load_model(self, file_name):
        self.model = joblib.load(file_name)
        print(f"Model loaded from {file_name}")

    @timer
    def grid_search(self, X_train, y_train, param_grid, cv=3):
        lgbm = lgb.LGBMRegressor(**self.params)
        grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.params.update(grid_search.best_params_)
        print(f"Best Parameters Found : {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
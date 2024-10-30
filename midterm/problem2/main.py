import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper

def read_X_train():
    file_path = f'./data/X_train.csv'
    X_train_df = pd.read_csv(file_path, header=0)
    return X_train_df

def read_X_test():
    file_path = f'./data/X_test.csv'
    X_test_df = pd.read_csv(file_path, header=0)
    return X_test_df

def read_y_train():
    file_path = f'./data/y_train.csv'
    y_train_df = pd.read_csv(file_path, header=0)
    return y_train_df

@timer
def rf_classifing_loo(X_train, y_train, X_test):
    rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=5 ,random_state=0)

    loo = LeaveOneOut()

    loo_y_true = []
    loo_y_pred = []

    for train_index, test_index in loo.split(X_train, y_train):
        X_train_fold = X_train.iloc[train_index]
        X_test_fold = X_train.iloc[test_index]
        y_train_fold = y_train.iloc[train_index]
        y_test_fold = y_train.iloc[test_index]

        rf_classifier.fit(X_train_fold, y_train_fold)
        loo_y_pred.append(rf_classifier.predict(X_test_fold)[0])
        loo_y_true.append(y_test_fold)

    f1 = f1_score(loo_y_true, loo_y_pred, average='binary')
    print(f"RF LOOCV F1 Score : {f1:.4f}")

    y_pred = rf_classifier.predict(X_test)
    print(f"RF Predicted Label for X_test Data: {y_pred}")

    return y_pred

@timer
def gbm_classifing_loo(X_train, y_train, X_test, n_estimators, learning_rate):
    gbm = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,max_depth=3, random_state=0)
    loo = LeaveOneOut()

    y_true = []
    y_pred = []

    for train_index, test_index in loo.split(X_train, y_train):
        X_train_fold = X_train.iloc[train_index]
        X_test_fold = X_train.iloc[test_index]
        y_train_fold = y_train.iloc[train_index]
        y_test_fold = y_train.iloc[test_index]

        gbm.fit(X_train_fold, y_train_fold)
        y_pred.append(gbm.predict(X_test_fold)[0])
        y_true.append(y_test_fold)

    f1 = f1_score(y_true, y_pred, average='binary')
    print(f"GBM LOOCV F1 Score of M = {n_estimators}, Nu = {learning_rate}: {f1:.4f}")

    y_pred = gbm.predict(X_test)
    print(f"GBM Predicted Label for X_test Data  of M = {n_estimators}, Nu = {learning_rate}: {y_pred}")

    return y_pred


def __main__():
    X_train = read_X_train()
    X_test = read_X_test()
    y_train = read_y_train()
    y_train = y_train.squeeze()
    print("All Data has been called")

    rf_pred = rf_classifing_loo(X_train, y_train, X_test)
    rf_pred = pd.DataFrame(rf_pred, columns=['Y'])
    rf_pred.to_csv('rf_pred.csv', index=False)

    n_estimators_list = [100, 200, 300]
    learning_rate_list = [0.01, 0.02, 0.05]
    for n_estimators in n_estimators_list:
        for learning_rate in learning_rate_list:
            gbm_pred = gbm_classifing_loo(X_train, y_train, X_test, n_estimators, learning_rate)
            gbm_pred = pd.DataFrame(gbm_pred, columns=['Y'])
            gbm_pred.to_csv(f'gbm_pred_{n_estimators}_{learning_rate}.csv', index=False)

if __name__ == "__main__":
    __main__()
import pandas as pd
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score, KFold

def gpr(X_data, y):
    model = GaussianProcessRegressor()
    model.fit(X_data, y)

    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    scores = cross_val_score(model, X_data, y, cv=kf, scoring='r2')

    print(f"5-fold CV R^2 score: {scores}")
    print(f"mean R^2 score: {np.mean(scores)}")
    print(f"std R^2 score: {np.std(scores)}")

    return np.mean(scores)

def __main__():
    # y data
    y = pd.read_csv('../check/y.csv')

    result_dict = {}

    # poly_removed X data
    print("Gaussian Process Regression with Poly removed")
    poly_removed_X = pd.read_csv('../check/X_poly_removed.csv')
    poly_removed_gpr = gpr(poly_removed_X, y)
    result_dict["Poly removed"] = poly_removed_gpr
    print("=" * 50)

    # linear PCA X data
    print("Gaussian Process Regression with Linear PCA")
    linear_pca_X = pd.read_csv('../check/linear_pca_X.csv')
    linear_pca_gpr = gpr(linear_pca_X, y)
    result_dict["Linear PCA"] = linear_pca_gpr
    print("=" * 50)

    # kernel PCA X data
    print("Gaussian Process Regression with Kernel PCA")
    kernel_pca_X = pd.read_csv('../check/kernel_pca_X.csv')
    kernel_pca_gpr = gpr(kernel_pca_X, y)
    result_dict["Kernel PCA"] = kernel_pca_gpr
    print("=" * 50)

    # GPR X data
    print("Gaussian Process Regression with Gaussian Random Projection")
    grp_X = pd.read_csv('../check/grp_X.csv')
    grp_gpr = gpr(grp_X, y)
    result_dict["Gaussian Random Projection"] = grp_gpr
    print("=" * 50)

    result = pd.DataFrame(result_dict)
    print(result)
    result.to_csv('../check/gaussian_process_regression.csv', index=False)

if __name__ == '__main__':
    __main__()
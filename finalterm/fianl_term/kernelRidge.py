import pandas as pd
import numpy as np

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score, KFold

def kernel_ridge(X_data, y):
    model = KernelRidge(kernel='rbf', gamma=0.5, degree=3, coef0=1)
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
    print("Kernel Ridge with Poly removed")
    poly_removed_X = pd.read_csv('../check/X_poly_removed.csv')
    poly_removed_kr = kernel_ridge(poly_removed_X, y)
    result_dict["Poly removed"] = poly_removed_kr
    print("=" * 50)

    # linear PCA X data
    print("Kernel with Linear PCA")
    linear_pca_X = pd.read_csv('../check/linear_pca_X.csv')
    linear_pca_kr = kernel_ridge(linear_pca_X, y)
    result_dict["Linear PCA"] = linear_pca_kr
    print("=" * 50)

    # kernel PCA X data
    print("Kernel Ridge with Kernel PCA")
    kernel_pca_X = pd.read_csv('../check/kernel_pca_X.csv')
    kernel_pca_kr = kernel_ridge(kernel_pca_X, y)
    result_dict["Kernel PCA"] = kernel_pca_kr
    print("=" * 50)

    # GPR X data
    print("Kernel Ridge with Gaussian Random Projection")
    grp_X = pd.read_csv('../check/grp_X.csv')
    grp_kr = kernel_ridge(grp_X, y)
    result_dict["Gaussian Random Projection"] = grp_kr
    print("=" * 50)

    result = pd.DataFrame(result_dict)
    print(result)
    result.to_csv('../check/kernel_ridge.csv', index=False)

if __name__ == '__main__':
    __main__()
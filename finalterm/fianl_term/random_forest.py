import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

def random_forest(X_data, y):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_data, y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

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
    print("Random Forest with Poly removed")
    poly_removed_X = pd.read_csv('../check/X_poly_removed.csv')
    poly_removed_rand_forest = random_forest(poly_removed_X, y)
    result_dict["Poly removed"] = poly_removed_rand_forest
    print("=" * 50)

    # linear PCA X data
    print("Random Forest with Linear PCA")
    linear_pca_X = pd.read_csv('../check/linear_pca_X.csv')
    linear_pca_rand_forest = random_forest(linear_pca_X, y)
    result_dict["Linear PCA"] = linear_pca_rand_forest
    print("=" * 50)

    # kernel PCA X data
    print("Random Forest with Kernel PCA")
    kernel_pca_X = pd.read_csv('../check/kernel_pca_X.csv')
    kernel_pca_rand_forest = random_forest(kernel_pca_X, y)
    result_dict["Kernel PCA"] = kernel_pca_rand_forest
    print("=" * 50)

    # GPR X data
    print("Random Forest with Gaussian Random Projection")
    grp_X = pd.read_csv('../check/grp_X.csv')
    grp_rand_forest = random_forest(grp_X, y)
    result_dict["Gaussian Random Projection"] = grp_rand_forest
    print("=" * 50)

    result = pd.DataFrame(result_dict)
    print(result)
    result.to_csv('../check/random_forest.csv', index=False)

if __name__ == '__main__':
    __main__()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold


class DynamicPLS():
    def __init__(self, target_variance=0.95):
        self.target_variance = target_variance
        self.n_components = None
        self.pls = None
        self.explained_variance_ratio_ = None
        self.scores_ = None

    def fit(self, X, y):
        self.n_components = self.get_optimal_n_components(X, y)
        print(f"Number of Selected Features: {self.n_components}")

        self.pls = PLSRegression(n_components=self.n_components)

        self.pls.fit(X, y)

        self.scores_ = self.pls.score(X, y)

    def transform(self, X, y):
        if self.pls is None:
            raise ValueError("PLS is not fitted yet. Use fit() method first.")
        return self.scores_

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

    def get_optimal_n_components(self, X, y, threshold=0.95):
        max_components = X.shape[1]
        pls = PLSRegression(n_components=max_components)
        pls.fit(X, y)
        eigenvalues = np.var(pls.transform(X), axis=0)

        ratios = eigenvalues / eigenvalues.sum()
        self.explained_variance_ratio_ = np.cumsum(ratios)

        plt.plot(range(1, max_components + 1), self.explained_variance_ratio_, marker='o')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold*100}% Threshold')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Eigenvalue Ratio for PLS Components')
        plt.legend()
        plt.grid(True)
        plt.show()

        optimal_n_components = np.argmax(self.explained_variance_ratio_ >= threshold) + 1

        return optimal_n_components

def pls(X_data, y, threshold=0.95):
    model = DynamicPLS(threshold)
    model.fit_transform(X_data, y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model.pls, X_data, y, cv=kf, scoring='r2')

    print(f"5-fold CV R^2 score: {scores}")
    print(f"mean R^2 score: {np.mean(scores)}")
    print(f"std R^2 score: {np.std(scores)}")

    return np.mean(scores), model

def __main__():
    # y data
    y = pd.read_csv('./check/y.csv')

    result_dict = {}

    # poly_removed X data
    print("PLS with Poly removed")
    poly_removed_X = pd.read_csv('./check/X_poly_removed.csv')
    poly_removed_pls = pls(poly_removed_X, y)
    result_dict["Poly removed"] = poly_removed_pls
    print("=" * 50)

    # linear PCA X data
    print("PLS with Linear PCA")
    linear_pca_X = pd.read_csv('./check/linear_pca_X.csv')
    linear_pca_pls = pls(linear_pca_X, y)
    result_dict["Linear PCA"] = linear_pca_pls
    print("=" * 50)

    # kernel PCA X data
    print("PLS with Kernel PCA")
    kernel_pca_X = pd.read_csv('./check/kernel_pca_X.csv')
    kernel_pca_pls = pls(kernel_pca_X, y)
    result_dict["Kernel PCA"] = kernel_pca_pls
    print("=" * 50)

    # GPR X data
    print("PLS with Gaussian Random Projection")
    grp_X = pd.read_csv('./check/grp_X.csv')
    grp_pls = pls(grp_X, y)
    result_dict["Gaussian Random Projection"] = grp_pls
    print("=" * 50)

    result = pd.DataFrame(result_dict)
    print(result)
    result.to_csv('./check/pls.csv', index=False)

if __name__ == '__main__':
    __main__()
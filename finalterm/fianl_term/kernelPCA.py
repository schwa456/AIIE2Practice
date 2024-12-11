from sklearn.decomposition import KernelPCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score


class DynamicKernelPCA:
    def __init__(self, kernel='poly', gamma=None, degree=3, coef0=1, alpha=1.0, target_variance=0.9):
        """
        Kernal PCA Class
        :param kernel: kernel used for fitting('linear, 'poly', 'rbf', 'sigmoid')
        :param gamma: variable for RBF function
        :param degree: degree of polynomial kernel for poly kernal
        :param coef0: degree of polynomial kernel for poly and sigmoid kernel
        """
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.target_variance = target_variance
        self.kpca = None
        self.eigenvaluess_ = None
        self.n_components_ = None

    def fit(self, X):
        """
        Learning Kernal PCA
        :param X:Data
        :return:
        """
        n_components = min(X.shape[0], X.shape[1])
        self.kpca = KernelPCA(
            n_components=n_components,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            alpha=self.alpha,
            fit_inverse_transform = True,
            n_jobs = -1
        )
        X_transformed = self.kpca.fit_transform(X)

        self.eigenvalues_ = np.var(X_transformed, axis=0)

        explained_variance_ratio = self.eigenvalues_ / np.sum(self.eigenvalues_)

        cumulative_variance = np.cumsum(explained_variance_ratio)
        print(cumulative_variance)

        self.n_components_ = np.searchsorted(cumulative_variance, self.target_variance) + 1

        print(f"Number of Selected Components: {self.n_components_}")

    def transform(self, X):
        if self.kpca is None:
            raise ValueError("Kernal PCA is not fitted yet.")
        return self.kpca.transform(X)[:, :self.n_components_]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def visualizing(self):
        eigenvalues = self.eigenvalues_
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_variance = np.cumsum(explained_variance_ratio)

        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Ratio')
        plt.grid(True)
        plt.show()


def __main__():
    """
    t = np.linspace(0, 2* np.pi, 100)
    x = t * np.cos(t)
    y = t * np.sin(t)
    data = np.vstack((x, y)).T
    """

    X_poly_removed = pd.read_csv('../check/X_poly_removed.csv')

    kpca = DynamicKernelPCA(kernel='rbf', gamma=15, target_variance=0.9)

    data_reduced = kpca.fit_transform(X_poly_removed)

    print(data_reduced)
    print(f"Reduced Data Size: {data_reduced.shape}")
"""
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_poly_removed[:, 0], X_poly_removed[:, 1], c='r', marker='o')
    plt.title("Original Data")

    # Kernel PCA 변환 데이터
    plt.subplot(1, 2, 2)
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c='b', marker='o')
    plt.title("Transformed Data (Kernel PCA)")

    plt.show()
    """

if __name__ == '__main__':
    __main__()
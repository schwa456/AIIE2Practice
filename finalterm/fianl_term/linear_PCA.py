import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt

from interaction_regression import *


class DynamicPCA:
    def __init__(self, target_variance=0.999):
        self.target_variance = target_variance
        self.n_components = None
        self.ipca = None
        self.explained_variance_ratio_ = None
        self.components_ = None
        self.scores_ = None

    def fit(self, data) -> None:
        """
        Learning IPCA using data, decide n_components dynamically
        :param data: data frame (sample x feature)
        :return: None
        """
        # Calculate explained variance ratio using initial IPCA
        ipca = IncrementalPCA()
        ipca.fit(data)

        # calculate cumulative variance
        self.explained_variance_ratio_ = ipca.explained_variance_ratio_
        cumulative_variance = np.cumsum(ipca.explained_variance_ratio_)
        print(cumulative_variance)

        # Finding latent number of components which is over threshold(target_variance)
        self.n_components = np.argmax(cumulative_variance >= self.target_variance) + 1
        print(f"Number of Selected Features: {self.n_components}")

        # Create IPCA instance again with selected components
        self.ipca = IncrementalPCA(n_components=self.n_components)
        self.ipca.fit(data)
        self.components_ = self.ipca.components_
        self.scores_ = self.ipca.transform(data)

    def transform(self, data) -> np.ndarray:
        """
        Decreasing dimension using learned PCA
        :param data: (numpy.ndarray) input data (sample x feature)
        :return: numpy.ndarray
        """
        if self.ipca is None:
            raise ValueError("IPCA is not fitted yet. Use fit() method first.")
        return self.ipca.transform(data)

    def fit_transform(self, data):
        """
        decreasing dimensions using learned PCA
        :param data: input data (sample x feature)
        :return: numpy.ndarray
        """
        self.fit(data)
        return self.scores_

def __main__():
    X_poly_removed = pd.read_csv('../check/X_poly_removed.csv')

    pca = DynamicPCA()
    data_reduced = pca.fit_transform(X_poly_removed)
    print(data_reduced)
    print(f"Reduced Data Size: {data_reduced.shape}")
    X_linear_pca = pd.DataFrame(data_reduced)
    X_linear_pca.to_csv('../check/linear_pca_X.csv')

if __name__ == '__main__':
    __main__()
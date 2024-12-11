import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt

from output_histogram import *
from small_dist_reduction import *
from label_encoding import *
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
    """
    y = get_output()

    outlier_boundary = get_outlier(y)
    outliers = get_number_of_outliers(y, outlier_boundary)

    y = get_filtered_y(y, outliers)

    filtering_idx = get_filtering_idx(outliers)

    X_data = get_X_data()
    filtered_X_data = get_filtered_X_data(X_data, filtering_idx)
    X_reduced = get_reduced_X(filtered_X_data, y)
    X_label_encoded = get_label_encoded_X(X_reduced)

    interaction_df = get_interaction_df(X_label_encoded)

    interaction_df.index = y.index

    linear_model, X_poly = interaction_linear_regression(interaction_df, y)

    X_poly_removed = remove_over_p(X_label_encoded, X_poly, linear_model)
    """

    X_poly_removed = pd.read_csv('../check/X_poly_removed.csv')

    pca = DynamicPCA()
    data_reduced = pca.fit_transform(X_poly_removed)
    print(data_reduced)
    print(f"Reduced Data Size: {data_reduced.shape}")

if __name__ == '__main__':
    __main__()
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt


class DynamicPCA:
    def __init__(self, target_variance=0.9, batch_size=20):
        self.target_variance = target_variance
        self.batch_size = batch_size
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
        ipca = IncrementalPCA(batch_size=self.batch_size)
        ipca.fit(data)

        # calculate cumulative variance
        self.explained_variance_ratio_ = ipca.explained_variance_ratio_
        cumulative_variance = np.cumsum(ipca.explained_variance_ratio_)
        print(cumulative_variance)

        # Finding latent number of components which is over threshold(target_variance)
        self.n_components = np.argmax(cumulative_variance >= self.target_variance) + 1
        print(f"Number of Selected Features: {self.n_components}")

        # Create IPCA instance again with selected components
        self.ipca = IncrementalPCA(n_components=self.n_components, batch_size=self.batch_size)
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

    def plot_scree(self) -> None:
        """
        visualizing Scree plot for each component's explained variance
        :return: None
        """
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(self.explained_variance_ratio_) + 1),
                 self.explained_variance_ratio_, marker='o')
        plt.title('Scree Plot')
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.show()

    def plot_score(self, pc1=1, pc2=2) -> None:
        """
        visualizing PC1 and PC2
        :param pc1: First principal component
        :param pc2: Second principal component
        :return: None
        """

        if self.scores_ is None:
            raise ValueError("Scores is not fitted yet. Use fit() method first.")

        if pc1 > self.scores_.shape[1] or pc2 > self.scores_.shape[1]:
            raise ValueError(f"선택한 주성분 번호가 유효하지 않습니다. 사용 가능한 주성분 개수는 {self.scores_.shape[1]}개입니다.")

        plt.figure(figsize=(8, 5))
        plt.scatter(self.scores_[:, pc1 - 1], self.scores_[:, pc2 - 1], c='b', marker='o')
        plt.title(f"Score Plot (PC{pc1} vs PC{pc2})")
        plt.xlabel('Principal Components 1')
        plt.ylabel('Principal Components 2')
        plt.show()

    def plot_loading(self, pc=1) -> None:
        """
        Visualizing selected PC's Loading
        :return: None
        """
        if self.components_ is None:
            raise ValueError("Components is not fitted yet. Use fit() method first.")

        plt.figure(figsize=(8, 5))
        plt.bar(np.arange(len(self.components_[pc - 1])), self.components_[pc-1])
        plt.title(f'Loading Plot PC{pc}')
        plt.xlabel('Feature Index')
        plt.ylabel('Loading Value')
        plt.show()

def __main__():
    """
    fdc_folder_path = './TSV_Etch_Dataset/25um/FDC_Data'
    fdc_file_list = get_file_list(fdc_folder_path)

    oes_folder_path = './TSV_Etch_Dataset/25um/OES_Data'
    oes_file_list = get_file_list(oes_folder_path)

    output_path = './TSV_Etch_Dataset/25um/X.csv'
    """
    X = np.random.rand(17, 300)
    print(X)
    pca = DynamicPCA(0.95)
    data_reduced = pca.fit_transform(X)
    print(data_reduced)
    print(f"Reduced Data Size: {data_reduced.shape}")

    pca.plot_scree()
    pca.plot_score(pc1=1, pc2=2)
    pca.plot_loading(pc=1)

if __name__ == '__main__':
    __main__()

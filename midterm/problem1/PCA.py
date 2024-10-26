import numpy as np
import matplotlib.pyplot as plt


def svd_execution(X):
    X_centered = X - np.mean(X, axis=0)

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    return U, S, Vt

def explained_ratio(S):
    ratio = (S ** 2) / np.sum(S ** 2)
    return ratio

def cumulative_ratio(explained_ratio):
    return np.cumsum(explained_ratio)

def number_component(cm_ratio, target_variance):
    n_components = np.argmax(cm_ratio >= target_variance) + 1
    return n_components

def visualizing(cm_ratio, target_variance, n_components):
    plt.plot(range(1, len(cm_ratio) + 1), cm_ratio, marker='o', linestyle='--', label='Cumulative Variance')
    plt.axhline(y=target_variance, color='r', linestyle='-', label=f'Target Variance ({target_variance * 100:.2f}%) Threshold')
    plt.axvline(x=n_components, color='g', linestyle='--', label=f'PC = {n_components}')
    plt.text(n_components, target_variance, f'PC = {n_components}', ha='center', va='bottom', fontsize=10)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance Plot')
    plt.grid(True)
    plt.legend()
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
    U, S, Vt = svd_execution(X)
    target_variance = 0.9
    ex_ratio = explained_ratio(S)
    cm_ratio = cumulative_ratio(ex_ratio)
    n_components = number_component(cm_ratio, target_variance)

    visualizing(cm_ratio, target_variance, n_components)


if __name__ == '__main__':
    __main__()

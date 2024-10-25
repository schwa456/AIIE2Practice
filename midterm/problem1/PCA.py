from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import dataPreprocessing
import numpy as np
import matplotlib.pyplot as plt

fdc_df = dataPreprocessing.FDCdata('FDC_25um_Fault_C4F8_90to85_SCCM.CSV')
oes_data = dataPreprocessing.OESdata('OES_25um_Fault_C4F8_90to85_SCCM.csv')

merged_data = dataPreprocessing.merge_FDC_and_OES_data(oes_data, fdc_df)

data_unfolded = dataPreprocessing.unfolding(merged_data)

# Data scaling standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_unfolded)

# execution PCA
pca = PCA()
pca.fit(X_scaled)

# calculate explained variance ratio of each PCs and cumulated variance ratio PCs
explained_variance_ratio = pca.explained_variance_ratio_
cumulated_variance_ratio = np.cumsum(explained_variance_ratio)

# Visualize cumulated variance ratio
plt.figure(figsize = (8, 5))
plt.plot(range(1, len(cumulated_variance_ratio) + 1), cumulated_variance_ratio, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance by Principal Components')
plt.grid(True)
plt.show()

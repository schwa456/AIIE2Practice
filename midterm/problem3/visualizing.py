import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

pio.renderers.default = 'browser'

fruit_data = np.array([
    [6, 7, 8], # 사과
    [6, 8, 7], # 배
    [10, 3, 10], # 수박
    [3, 10, 2], # 체리
    [8, 5, 9] # 망고
])


column_means = fruit_data.mean(axis=0)

X = fruit_data - column_means
XXT = np.matmul(X, X.T)
np.set_printoptions(suppress=True, precision=4)

# 고유값 분해 수행
eigenvalues, eigenvectors = np.linalg.eigh(XXT)

# 고유값과 고유벡터를 큰 순서대로 정렬
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("고유값:\n", eigenvalues)
print("고유벡터:\n", eigenvectors)


k = 2

Lambda_k = np.diag(np.sqrt(eigenvalues[:k]))
Q_k = eigenvectors[:, :k]

print("Lambda_k : ", Lambda_k)
print("Q_k : ", Q_k)

Y = np.matmul(Q_k, Lambda_k)
print("Y : ", Y)

YYT = np.matmul(Y, Y.T)

print("YYT : ", YYT)


difference_each = XXT - YYT
print("difference_each : ", difference_each)
difference_fro = np.linalg.norm(XXT - YYT, ord='fro')
print("difference_fro : ", difference_fro)
relative_fro_norm = difference_fro / np.linalg.norm(XXT, ord='fro')
print("relative_fro_norm: ", relative_fro_norm)



"""

fruit_names = ['Apple', 'Pear', 'Watermelon', 'Cherry', 'Mango']

trace = go.Scatter3d(
    x=fruit_data[:, 0],
    y=fruit_data[:, 1],
    z=fruit_data[:, 2],
    mode='markers+text',
    marker=dict(size=5, color='green'),
    text=fruit_names,
    textposition='top center',
)

layout = go.Layout(
    title='3D Fruit Data Scatter Plot',
    scene=dict(
        xaxis=dict(title='Size'),
        yaxis=dict(title='Color'),
        zaxis=dict(title='Sweetness'),
    )
)

fig = go.Figure(data=[trace], layout=layout)
fig.show()

mds =MDS(n_components=2, random_state=42)
fruit_2d = mds.fit_transform(fruit_data)

plt.scatter(fruit_2d[:, 0], fruit_2d[:, 1], color='green')
for i, (x, y) in enumerate(fruit_2d):
    plt.text(x, y, fruit_names[i], fontsize=12)
plt.title('MDS Projection of Fruits (3D -> 2D)')
plt.show()
"""
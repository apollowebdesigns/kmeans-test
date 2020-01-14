import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
import matplotlib.markers

# Create data
from scipy.spatial.distance import cdist

N = 500
x = np.random.rand(N) * 500
y = np.random.rand(N) * 500
colors = (0, 0, 0)
area = np.pi * 3
plt.axis("off")
# Plot
plt.scatter(x, y, s=area * (4 * np.sin(x / 25) * np.sin(y / 25)), c='#40e0d0', alpha=1)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('test.png')

#### Kmeans

from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

Data = {
    'x': x,
    'y': y
}

df = DataFrame(Data, columns=['x', 'y'])

kmeans = KMeans(n_clusters=18).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

plt.savefig('test3.png')


def elbow():
    from sklearn.metrics import silhouette_score

    sil = []
    kmax = 10

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(df)
        labels = kmeans.labels_
        sil.append(silhouette_score(df, labels, metric='euclidean'))
    return sil


print(elbow())

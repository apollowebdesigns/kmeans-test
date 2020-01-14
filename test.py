import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import operator

PINK = 'rgb(214, 121, 186)'

data = {'x': [], 'y': []}


def run():
    img = cv2.imread('circle.png')
    actual = img.copy()
    img[np.where((img != [255, 0, 255]).all(axis=2))] = [0, 0, 0]
    img[np.where((img == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
    img[np.where((img == [255, 0, 255]).all(axis=2))] = [255, 255, 255]
    cv2.imwrite("debug.jpg", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    print(n_labels)

    size_thresh = 1
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= size_thresh:
            # print(stats[i, cv2.CC_STAT_AREA])
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
            # cv2.rectangle(actual, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
            # TODO needs to be rectangular!
            for i in np.arange(x, x+w, 0.1):
                data['x'].append(x+i)
                data['y'].append(y+i)


    df = pd.DataFrame(data, columns=['x', 'y'])
    n_clusters = 3

    res = _find_optimum_clusters(df)
    k_means = KMeans(n_clusters=n_clusters).fit(df)
    centroids_ = k_means.cluster_centers_
    print(centroids_)

    plt.scatter(df['x'], df['y'], c=k_means.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(centroids_[:, 0], centroids_[:, 1], c='red', s=50)
    plt.savefig('centroids.png')
    x_coordinates = [i for i in centroids_[:, 0]]
    y_coordinates = [i for i in centroids_[:, 1]]
    w = 4
    h = 4
    for i, x in enumerate(x_coordinates):
        # cv2.rectangle(img, (x - w/2, y - h/2), (x + w/2, y + h/2), (0, 255, 0), thickness=5)
        cv2.rectangle(actual, (int(x), int(y_coordinates[i])), (int(x + w), int(y_coordinates[i] + h)), (0, 255, 0), thickness=20)
    cv2.imwrite("out.jpg", actual)


# def _find_optimum_clusters(x):
#     res = list()
#     n_cluster = range(2, 20)
#     for n in n_cluster:
#         kmeans = KMeans(n_clusters=n)
#         kmeans.fit(x)
#         res.append(np.average(np.min(cdist(x, kmeans.cluster_centers_, 'euclidean'), axis=1)))
#     plt.plot(n_cluster, res)
#     plt.title('elbow curve')
#     plt.savefig('f.png')
#     return res

def _find_optimum_clusters(x):
    distorsions = []
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(x)
        distorsions.append(kmeans.inertia_)
    plt.plot(range(2, 20), distorsions)
    plt.title('elbow curve')
    plt.savefig('f.png')
    index, value = max(enumerate(distorsions), key=operator.itemgetter(1))
    print(f'the max distortion is:{value} at index {index}')
    return distorsions


if __name__ == "__main__":
    run()
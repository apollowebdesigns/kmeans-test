import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import operator

PINK = 'rgb(214, 121, 186)'

from kneed import KneeLocator


def run():
    MEDIAN = 36
    data = {'x': [], 'y': []}
    img = cv2.imread('test.png')
    actual = img.copy()
    img[np.where((img == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
    img[np.where((img != [64, 224, 208]).all(axis=2))] = [0, 0, 0]
    img[np.where((img != [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    # img[np.where((img == [64, 224, 208]).all(axis=2))] = [255, 255, 255]
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
            step_weight = 1
            areas = w*h
            if areas > MEDIAN * 10:
                for x_value in np.arange(x, x+w, step_weight):
                    for y_value in np.arange(y, y+h, step_weight):
                        data['x'].append(x_value)
                        data['y'].append(y_value)

    df = pd.DataFrame(data, columns=['x', 'y'])
    print('max values are', max(data['x']))
    print('max values are', df.max())
    n_clusters = 18

    n_clusters = _find_optimum_clusters(df, 50)


    k_means = KMeans(n_clusters=n_clusters).fit(df)
    centroids_ = k_means.cluster_centers_
    print(centroids_)



    plt.scatter(df['x'], df['y'], c=k_means.labels_.astype(float), s=50, alpha=0.5)
    plt.savefig('debug graph.png')
    plt.clf()
    plt.scatter(centroids_[:, 0], centroids_[:, 1], c='red', s=50)
    plt.savefig('centroids.png')
    plt.clf()
    x_coordinates = [i for i in centroids_[:, 0]]
    y_coordinates = [i for i in centroids_[:, 1]]
    w = 4
    h = 4
    for i, x in enumerate(x_coordinates):
        # cv2.rectangle(img, (x - w/2, y - h/2), (x + w/2, y + h/2), (0, 255, 0), thickness=5)
        cv2.rectangle(actual, (int(x), int(y_coordinates[i])), (int(x + w), int(y_coordinates[i] + h)), (0, 255, 0),
                      thickness=20)
    cv2.imwrite("out.jpg", actual)


def _find_optimum_clusters(x, max_cluster_number):
    step_size = 1
    distorsions = []
    for i, k in enumerate(range(2, max_cluster_number, step_size)):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(x)
        distorsions.append(kmeans.inertia_)
    kn = KneeLocator(range(2, max_cluster_number, step_size), distorsions, curve='convex', direction='decreasing')
    print('optimum is')
    print(kn.knee)
    plt.plot(range(2, max_cluster_number, step_size), distorsions)
    plt.title('elbow curve')
    plt.savefig('f.png')
    plt.clf()
    index, value = max(enumerate(distorsions), key=operator.itemgetter(1))
    print(f'the max distortion is:{value} at index {index}')
    return kn.knee


if __name__ == "__main__":
    run()

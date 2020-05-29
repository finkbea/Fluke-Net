from sklearn.cluster import KMeans
import numpy as np
matplotlib.use('TkAgg')
from sklearn.metrics import silhouette_score

def kMeans(x, k):
    #to be used once the optimal k (n_clusters) is found
    kmeans = KMeans(n_clusters=k).fit(x)
    print('labels ',kmeans.labels_)
    kmeans.predict(x)
    print('cluster centers ',kmeans.cluster_centers_)
    
    plt.scatter(x[:, 0], x[:, -1])
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', marker='x')
    plt.title('Kmeans cluster')
    plt.show()
    
def findK(x):
    #silhouette_score is between -1 and 1, higher value means better matched to its cluster
    clusterRange = list (range(2,10))

    for n in clusterRange:
        kmeans = KMeans(n_clusters=n)
        preds = kmeans.fit_predict(x)
        centers = kmeans.cluster_centers_

        score = silhouette_score(x, preds)
        print("For n_clusters = {}, silhouette score is {})".format(n, score))

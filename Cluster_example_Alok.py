# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 17:53:30 2020

@author: Alok Ranjan
@Company Name: Nikhil Analytics
@Official email id: alokranjan@nikhilanalytics.com; dyutilal@nikhilanalytics.com
@Personal email id: alokranjan1502@gmail.com; dyutilal@gmail.com
@WattsApps Number: +91-9741267715/+91-9886972051/+91-9945339324

Clustering Logic

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('E:/Alok/Business/Upwork/Multiple High End Technology Content/Clustering/Python Code and dataset/Mall_Customers.csv')
dataset.head()
dataset.shape

X = dataset.iloc[:, [3, 4]].values
import scipy.cluster.hierarchy as sch
dendrogrm = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()

#As we have already discussed to choose the number of clusters we 
#draw a horizontal line to the longest line that traverses maximum
# distance up and down without intersecting the merging points. 
#So we draw a horizontal line and the number of verticle lines it 
#intersects is the optimal number of clusters.

#In this case, it's 5. So let's fit our Agglomerative model with 
#5 clusters.

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters 
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 50, c = 'red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 50, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 50, c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 50, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 50, c = 'magenta', label = 'Sensible')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# =============================================================================
# Kmeans
# =============================================================================

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# =============================================================================
# GMM
# =============================================================================
import os
os.chdir("E:/Alok/Business/Upwork/Multiple High End Technology Content/Clustering/Python Code and dataset")
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Clustering_gmm.csv')
data.shape
# training gaussian mixture model 
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4)
gmm.fit(data)

#predictions from gmm
labels = gmm.predict(data)
frame = pd.DataFrame(data)
frame['cluster'] = labels
frame.columns = ['Weight', 'Height', 'cluster']

color=['blue','green','cyan', 'black']
for k in range(0,4):
    data = frame[frame["cluster"]==k]
    plt.scatter(data["Weight"],data["Height"],c=color[k])
plt.show()
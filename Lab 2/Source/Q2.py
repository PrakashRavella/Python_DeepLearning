# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import pca
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import *
from IPython.display import display
import random
import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import silhouette_score

heart = pd.read_csv('heart.csv')

heart.info()

heart.describe()

print('Check if any column have missing value', heart.isnull().sum())

for i in heart.columns:
    plt.figure(figsize=(7, 6))
    heart[i].hist()
    plt.xlabel(str(i))
    plt.ylabel("freq")

heart.drop('thalach', axis=1, inplace=True)

corrmat = heart.corr(method='spearman')
f, ax = plt.subplots(figsize=(8, 8))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaler.fit(heart)
X_scaled_array = scaler.transform(heart)
X_scaled = pd.DataFrame(X_scaled_array, columns=heart.columns)

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_scaled)
# cluster centers
print(kmeans.cluster_centers_)

WCSS = []
##elbow method to know the number of clusters
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=20, random_state=5)
    kmeans.fit(X_scaled)
    cluster_an = kmeans.predict(X_scaled)
    WCSS.append(kmeans.inertia_)
    plt.scatter(X_scaled_array[:, 0], X_scaled_array[:, 3], c=cluster_an, s=20)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
    plt.show()
    s = silhouette_score(X_scaled, cluster_an, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(i, s))

plt.plot(range(2, 11), WCSS)
plt.title('elbow method')
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()


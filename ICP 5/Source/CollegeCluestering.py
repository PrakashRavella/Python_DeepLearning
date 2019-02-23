import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("College.csv")
X_data = data.iloc[:,2:]
print(X_data.columns)
nclusters=3
seed=7
km=KMeans(n_clusters=nclusters,random_state=seed)
km.fit(X_data)
y_cluster_kmeans=km.predict(X_data)
print(y_cluster_kmeans)

plt.show()
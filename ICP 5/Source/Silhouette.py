from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score

dataset = pd.read_csv('College.csv')
keep_col = ['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad', 'P.Undergrad', 'Outstate',
                'Room.Board', 'Books', 'Personal', 'PhD', 'Terminal', 'S.F.Ratio', 'perc.alumni', 'Expend', 'Grad.Rate']
data = dataset[keep_col]
data.to_csv("sFile.csv", index=False)
for i in range(2, 7):
        kmean = KMeans(n_clusters=i, max_iter=300)
        x = kmean.fit_predict(data)
        # x=kmean.predict(test)
        sil = silhouette_score(data, x)
        print("for cluster ", i, "silhouette score is", sil)


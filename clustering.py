import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist
import warnings

# 1. Membaca dataset dan menampilkan data
dataset = pd.read_csv('heart.csv')
print("Dataset:")
print(dataset.head())

# 2. Normalisasi data dengan min-max (0-1)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(dataset)
normalized_df = pd.DataFrame(normalized_data, columns=dataset.columns)
print("\nData setelah dinormalisasi:")
print(normalized_df.head())

# 3. Melakukan clustering menggunakan K-means (k=2)
kmeans = KMeans(n_clusters=2, random_state=0 , n_init=10)
kmeans.fit(normalized_df)
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
print("\nHasil clustering dengan K-means (k=2):")
print(cluster_labels)

# 4. Melakukan clustering dengan Single, Average, dan Complete Linkage (k=2)
linkage_methods = ['single', 'average', 'complete']
for method in linkage_methods:
    linkage_clusters = AgglomerativeClustering(n_clusters=2, linkage=method)
    linkage_labels = linkage_clusters.fit_predict(normalized_df)
    print(f"\nHasil clustering dengan {method.capitalize()} Linkage (k=2):")
    print(linkage_labels)

# 5. Melakukan clustering dengan atribut yang paling berpengaruh menggunakan K-Means (k=3) sebanyak 10 kali
num_attributes = 10
cluster_i = []
cluster_val = []

for i in range(1, num_attributes + 1):
    attributes = normalized_df.nlargest(i, 'target').columns
    selected_data = normalized_df[attributes]
    
    sse_values = []
    for _ in range(10):
        kmeans = KMeans(n_clusters=3, random_state=0 , n_init=10)
        kmeans.fit(selected_data)
        sse = np.sum(np.min(cdist(selected_data, kmeans.cluster_centers_, 'euclidean'), axis=1))
        sse_values.append(sse)
    
    best_sse_index = np.argmin(sse_values)
    cluster_i.append(best_sse_index + 1)
    cluster_val.append(sse_values[best_sse_index])

print("\nHasil clustering dengan atribut yang paling berpengaruh menggunakan K-Means (k=3):")
print("cluster_i:", cluster_i)
print("cluster_val:", cluster_val)

# 6. Mengambil cluster_i dengan cluster_val terkecil
smallest_val_index = np.argmin(cluster_val)
smallest_cluster_i = cluster_i[smallest_val_index]
print("\nCluster_i dengan cluster_val terkecil:", smallest_cluster_i)

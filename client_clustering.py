# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 21:48:21 2025

This file uses python's scikit-learn library to create client clusters for 
efficient dividing up the client sales calls among salesmen


@author: josh_
"""

import pandas as pd

df = pd.read_csv(".\potential_clients.csv")

df.head()

import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

plt.figure(1)
plt.scatter(df['x'], df['y'], s=5)
plt.title('Potential Clients')

scaler = StandardScaler()

df_scaled = scaler.fit_transform(df)

variations = []
range_n_clusters = range(2,20)
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(df_scaled)
    
    variations.append(kmeans.inertia_)

plt.figure(2)
plt.plot(range_n_clusters, variations, 'bx-')
plt.xlabel('Number of Clusters(k)')
plt.ylabel('Variation')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# Silhouette analysis
range_n_clusters = range(2,20)

optimal_num_clusters = 0
max_silhouette_score = 0

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(df_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(df, cluster_labels)
    
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    
    if silhouette_avg > max_silhouette_score:
        max_silhouette_score = silhouette_avg
        optimal_num_clusters=num_clusters
    
print("Optimal number of clusters = {}".format(optimal_num_clusters))


kmeans = KMeans(n_clusters=optimal_num_clusters, max_iter=50)
kmeans.fit(df_scaled)

plt.figure(3)
centroids = kmeans.cluster_centers_
plt.scatter(df_scaled[:, 0], df_scaled[:, 1], s=5, c=kmeans.labels_, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='red', s=70, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Potential Client Groups')


# TRY THIS AGAIN WITH FEWER CLUSTERS

variations = []
range_n_clusters = range(2,10)
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(df_scaled)
    
    variations.append(kmeans.inertia_)

plt.figure(4)
plt.plot(range_n_clusters, variations, 'bx-')
plt.xlabel('Number of Clusters(k)')
plt.ylabel('Variation')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# Silhouette analysis
range_n_clusters = range(2,6)

optimal_num_clusters = 0
max_silhouette_score = 0

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(df_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(df, cluster_labels)
    
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    
    if silhouette_avg > max_silhouette_score:
        max_silhouette_score = silhouette_avg
        optimal_num_clusters=num_clusters
    
print("Optimal number of clusters = {}".format(optimal_num_clusters))


kmeans = KMeans(n_clusters=optimal_num_clusters, max_iter=50)
kmeans.fit(df_scaled)

plt.figure(5)
centroids = kmeans.cluster_centers_
plt.scatter(df_scaled[:, 0], df_scaled[:, 1], s=5, c=kmeans.labels_, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='red', s=70, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Potential Client Groups')




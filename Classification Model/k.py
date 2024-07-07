import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use an interactive backend
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

state = pd.read_csv('modified_clustering_data.csv')

points = state[['Latitude', 'Longitude']].values


class KMeansCLustering:

    def __init__(self,k=3):
        self.k=k
        self.centroids = None

    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2,axis=1))

    
    def fit(self, X, max_iterations=200):
        self.centroids=np.random.uniform(np.amin(X, axis=0), np.amin(X, axis=0),size=(self.k, X.shape[1]))##X.shape[1]:will give the number of columns
        
        for num in range(max_iterations):
            y = []

            for data_point in X:
                distances = KMeansCLustering.euclidean_distance(data_point, self.centroids)
                # calcute minimum of the distances and then give the index of the minimum to y
                cluster_num = np.argmin(distances) # it passes the index
                y.append(cluster_num)

            
            y = np.array(y)

            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i)) ## has all the indices where i equals the value in y

            cluster_centres = []

            for i, indices in enumerate(cluster_indices):
                if len( indices )==0 :
                    cluster_centres.append(self.centroids[i])

                else:
                    cluster_centres.append(np.mean(X[indices],axis = 0)[0])

            if np.max(self.centroids- np.array(cluster_centres)) < 0.00001:
                break
            else :
                self.centroids = np.array(cluster_centres)

        return y


kmeans = KMeansCLustering(k=3)

labels = kmeans.fit(points)

plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')  # Adjust 'cmap' as desired
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker="*", s=200)
plt.show()

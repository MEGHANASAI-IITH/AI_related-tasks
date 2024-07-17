
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df_pincodes = pd.read_csv('modified_clustering_data.csv')

# Select relevant columns (longitude and latitude)
points = df_pincodes[['Longitude', 'Latitude']].values

# Define a range of k values to test
k_values = range(2, 33)  # Adjust the range as needed

# Initialize list to store inertia values
inertia_values = []

# Function to calculate the inertia (sum of squared distances to the nearest centroid)
def calculate_inertia(points, centroids, labels):
    distances = np.sqrt(((points - centroids[labels])**2).sum(axis=1))
    return np.sum(distances)

# K-means algorithm to find the optimal centroids and labels
def kmeans(points, k, max_iterations=100):
    # Randomly initialize the centroids
    centroids = points[np.random.choice(points.shape[0], k, replace=False)]
    for _ in range(max_iterations):
        # Calculate distances from points to centroids
        distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
        # Assign each point to the nearest centroid
        labels = np.argmin(distances, axis=0)
        # Calculate new centroids as the mean of the points in each cluster
        new_centroids = np.array([points[labels == i].mean(axis=0) for i in range(k)])
        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Calculate inertia for each k
for k in k_values:
    centroids, labels = kmeans(points, k)
    inertia = calculate_inertia(points, centroids, labels)
    inertia_values.append(inertia)

# Plotting the Elbow Method (Inertia)
plt.figure(figsize=(10, 5))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.xticks(k_values)
plt.grid(True)
plt.show()
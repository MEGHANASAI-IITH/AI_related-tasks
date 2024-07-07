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

# Calculate inertia for each k
for k in k_values:
    centroids = points[np.random.choice(points.shape[0], k, replace=False)]
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    inertia = sum(np.min(distances, axis=0))
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

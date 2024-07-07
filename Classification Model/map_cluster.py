import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the shapefile
shapefile_path = 'TS_District_Boundary_33_FINAL.shp'
gdf = gpd.read_file(shapefile_path)

# Reproject the shapefile to WGS 84 (latitude and longitude)
gdf = gdf.to_crs("EPSG:4326")

# Load the pincode data from a CSV file
pincode_data_path = 'modified_clustering_data.csv'
df_pincodes = pd.read_csv(pincode_data_path)

# Convert the dataframe to a GeoDataFrame with WGS 84 CRS
gdf_pincodes = gpd.GeoDataFrame(
    df_pincodes, 
    geometry=gpd.points_from_xy(df_pincodes.Longitude, df_pincodes.Latitude),
    crs="EPSG:4326"
)

# Perform a spatial join to filter pincodes within the district boundaries
gdf_pincodes_within_bounds = gpd.sjoin(gdf_pincodes, gdf, how="inner")

# Debugging: Print CRS and first few rows of data
print("Shapefile CRS:", gdf.crs)
print("Pincode GeoDataFrame CRS:", gdf_pincodes.crs)
print("Filtered Pincodes GeoDataFrame:")
print(gdf_pincodes_within_bounds.head())

# Perform K-means clustering on Longitude and Latitude
points = gdf_pincodes_within_bounds[['Longitude', 'Latitude']].values

class KMeansCLustering:
    def __init__(self, k=20):
        self.k = k
        self.centroids = None
    
    def euclidean_distance(self, data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))
    
    def fit(self, X, max_iterations=200):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]  # Initialize centroids randomly
        
        for _ in range(max_iterations):
            distances = np.array([self.euclidean_distance(point, self.centroids) for point in X])
            labels = np.argmin(distances, axis=1)  # Assign points to closest centroid
            
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            
            if np.allclose(self.centroids, new_centroids, rtol=1e-6):
                break
            
            self.centroids = new_centroids
        
        return labels

# Initialize and fit K-means clustering
kmeans = KMeansCLustering(k=20)
labels = kmeans.fit(points)

# Plot the district boundaries and clustered points
fig, ax = plt.subplots(figsize=(12, 10))

# Plot district boundaries
gdf.plot(ax=ax, color='lightgrey', edgecolor='black')

# Plot clustered points
scatter = ax.scatter(gdf_pincodes_within_bounds['Longitude'], 
                     gdf_pincodes_within_bounds['Latitude'], 
                     c=labels, cmap='viridis', s=50, alpha=0.8, label='Clustered Points')

# Plot centroids
ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='*', s=200, label='Centroids')

# Set limits to ensure all points are visible
xmin, ymin, xmax, ymax = gdf.total_bounds
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Add title and labels
plt.title('District Boundaries with Clustered Points')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Add legend
legend1 = ax.legend(*scatter.legend_elements(), loc="lower right", title="Clusters")
legend2 = ax.legend(handles=[scatter], loc="upper right")
ax.add_artist(legend1)

# Show the plot
plt.show()

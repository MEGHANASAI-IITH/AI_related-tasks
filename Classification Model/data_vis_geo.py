import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

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

# Print bounds of the shapefile and filtered pincode points
print("Shapefile bounds:", gdf.total_bounds)
print("Filtered Pincode points bounds:", gdf_pincodes_within_bounds.total_bounds)

# Plot the shapefile and filtered pincode locations
fig, ax = plt.subplots(figsize=(10, 10))

# Plot district boundaries
gdf.plot(ax=ax, color='lightgrey', edgecolor='black')

# Plot filtered pincode locations
gdf_pincodes_within_bounds.plot(ax=ax, color='red', markersize=50, label='Pincode Locations', alpha=0.6)

# Set limits to ensure all points are visible
xmin, ymin, xmax, ymax = gdf.total_bounds
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Add title and labels
plt.title('District Boundaries and Pincode Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()

# Show the plot
plt.show()

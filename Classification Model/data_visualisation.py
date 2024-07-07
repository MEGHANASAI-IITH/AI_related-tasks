from geopy.geocoders import ArcGIS
import pandas as pd
import folium

# Load the data
state = pd.read_csv('modified_clustering_data.csv')

# Drop rows with NaN values in 'Latitude' or 'Longitude'
state = state.dropna(subset=['Latitude', 'Longitude'])

# Extract latitude and longitude values
lst = state[['Latitude', 'Longitude']].values.tolist()

# Create a map object centered around the first valid location
location_map = folium.Map(location=[lst[0][0], lst[0][1]], zoom_start=5)

# Create a FeatureGroup
fg = folium.FeatureGroup(name='pincode')

# Add markers to the FeatureGroup
for i in lst:
    fg.add_child(folium.Marker(location=[i[0], i[1]], icon=folium.Icon(color='blue')))

# Add the FeatureGroup to the map
location_map.add_child(fg)

# Save the map to an HTML file
map_filename = 'map.html'
location_map.save(map_filename)



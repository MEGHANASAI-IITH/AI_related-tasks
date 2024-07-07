import pandas as pd

# Load the data with low_memory set to False to avoid DtypeWarning
df = pd.read_csv('clustering_data.csv', low_memory=False)

# Convert 'Latitude' to numeric, coercing errors to NaN
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

# Filter the DataFrame to include only rows where pincode is starting with 50 and Latitude is between 15 and 20
df_filtered = df[(df['Pincode'].astype(str).str.startswith('50')) & 
                 (df['Latitude'].between(15, 20, inclusive='neither')) & 
                 (df['Longitude'].between(77, 82, inclusive='neither'))]

# Drop identical rows
df_filtered = df_filtered.drop_duplicates()

# Drop rows with NaN values
df_filtered = df_filtered.dropna()

# Save the filtered data to a new CSV file
df_filtered.to_csv('modified_clustering_data.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt

# Adjust chunk size and dtype as needed
chunk_size = 1000
chunks = pd.read_csv('alphabets.csv', chunksize=chunk_size, low_memory=False)

# Initialize lists to store chunks
x_chunks = []
y_chunks = []

# Process each chunk
for chunk in chunks:
    # Skip any non-data rows (assuming there might be headers or metadata)
    chunk = chunk.dropna()  # Drop rows with NaN values
    
    # Extract features (assuming features start from the second column)
    x_chunk = chunk.iloc[:, 1:].values.astype(np.float32)  # Features
    
    # Extract labels (assuming the label column is named 'label')
    y_chunk = chunk['label'].values  # Labels remain as strings or categorical

    x_chunks.append(x_chunk)
    y_chunks.append(y_chunk)

# Concatenate all chunks
x = np.concatenate(x_chunks, axis=0)
y = np.concatenate(y_chunks, axis=0)

# Normalize x
x_normalized = x / 255.0 

# Reshape x to match the input shape expected by the first layer of your model
x_reshaped = x_normalized.reshape(-1, 28, 28)

# Encode labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x_reshaped, y_encoded, test_size=0.2, random_state=42)

model = tf.keras.models.load_model('Handwritten_model.h5')
loss, accuracy = model.evaluate(x_test,y_test)

print(loss)
print(accuracy)
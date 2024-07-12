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
chunks = pd.read_csv('alphabets_28x28.csv', chunksize=chunk_size, low_memory=False)

# Initialize lists to store chunks
x_chunks = []
y_chunks = []

# Process each chunk
for chunk in chunks:
     chunk = chunk.dropna()
     
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

# Save the label encoder
np.save('label_classes.npy', label_encoder.classes_)

x_train, x_test, y_train, y_test = train_test_split(x_reshaped, y_encoded, test_size=0.2, random_state=42)



model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(26, activation='softmax'))  # Assuming 26 letters

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('Handwritten_model.h5')

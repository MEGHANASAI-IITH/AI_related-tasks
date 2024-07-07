import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import csv

# Load the trained model
model = tf.keras.models.load_model('Handwritten_model.h5')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_classes.npy', allow_pickle=True)

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return []

    height, width = img.shape
    print(f"Image shape: {img.shape}")

    # List to store all 28x28 squares
    squares = []

    for y in range(0, height, 28):
        for x in range(0, width, 28):
            square = img[y:y+28, x:x+28]
            if square.shape == (28, 28):  # Ensure the square is 28x28 pixels
                squares.append(square)

    return squares

def predict_letter(letter_img):
    letter_img = letter_img.reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(letter_img)
    predicted_label = np.argmax(prediction, axis=1)
    return label_encoder.inverse_transform(predicted_label)[0]

def read_image(image_path):
    squares = preprocess_image(image_path)
    if not squares:
        print("No squares detected")
        return []

    recognized_letters = []
    for square in squares:
        if np.all(square == 0):  # Check if all pixels are zero
            recognized_letters.append(' ')
        else:
            letter = predict_letter(square)
            recognized_letters.append(letter)
    
    return recognized_letters

def save_to_csv(line, csv_path):
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([line])

csv_path = 'recognized_lines.csv'

# Clear the CSV file if it exists
if os.path.exists(csv_path):
    os.remove(csv_path)

# Example usage
image_number = 1
while os.path.isfile(f"target_images/line_{image_number}.png"):
    image_path = f"target_images/line_{image_number}.png"
    recognized_letters = read_image(image_path)
    recognized_line = ''.join(recognized_letters)
    print(f"Recognized Line {image_number}: {recognized_line}")
    save_to_csv(recognized_line, csv_path)
    image_number += 1




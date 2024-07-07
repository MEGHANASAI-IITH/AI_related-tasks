import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import csv
import re
import pandas as pd
import math
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv('sentiment_analysis_dataset.csv')

# Function to remove HTML tags and URLs
def remove_tags(string):
    result = re.sub(r'<.*?>', '', string)          # remove HTML tags
    result = re.sub(r'https?://\S+|www\.\S+', '', result)   # remove URLs
    result = re.sub(r'[^a-zA-Z\s]', '', result)    # remove non-alphanumeric characters 
    result = result.lower()
    return result

# Apply the function to the line column
data['line'] = data['line'].apply(lambda cw: remove_tags(cw))

# Import stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
data['line'] = data['line'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Tokenize and lemmatize
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

data['line'] = data['line'].apply(lemmatize_text)

# Prepare data for model
lines = data['line'].values
labels = data['sentiment'].values
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Split the data
train_lines, test_lines, train_labels, test_labels = train_test_split(lines, encoded_labels, stratify=encoded_labels, test_size=0.2, random_state=42)

# Vectorize the data
vec = CountVectorizer(max_features=3000)
X_train = vec.fit_transform(train_lines).toarray()
X_test = vec.transform(test_lines).toarray()
vocab = vec.get_feature_names_out()

# Initialize word counts
word_counts = {label: defaultdict(int) for label in np.unique(train_labels)}
for i in range(X_train.shape[0]):
    label = train_labels[i]
    for j in range(len(vocab)):
        word_counts[label][vocab[j]] += X_train[i][j]

# Laplace smoothing function
def laplace_smoothing(n_label_items, vocab, word_counts, word, text_label):
    a = word_counts[text_label][word] + 1
    b = n_label_items[text_label] + len(vocab)
    return math.log(a / b)

# Group data by label
def group_by_label(x, y, labels):
    data = {}
    for l in labels:
        data[l] = x[np.where(y == l)]
    return data

# Fit function
def fit(x, y, labels):
    n_label_items = {}
    log_label_priors = {}
    n = len(x)
    grouped_data = group_by_label(x, y, labels)
    for l, data in grouped_data.items():
        n_label_items[l] = len(data)
        log_label_priors[l] = math.log(n_label_items[l] / n)
    return n_label_items, log_label_priors

# Predict function
def predict(n_label_items, vocab, word_counts, log_label_priors, labels, x):
    result = []
    for text in x:
        label_scores = {l: log_label_priors[l] for l in labels}
        words = set(w_tokenizer.tokenize(text))
        for word in words:
            if word not in vocab:
                continue
            for l in labels:
                log_w_given_l = laplace_smoothing(n_label_items, vocab, word_counts, word, l)
                label_scores[l] += log_w_given_l
        result.append(max(label_scores, key=label_scores.get))
    return result

# Define labels
labels = np.unique(train_labels)

# Fit the model
n_label_items, log_label_priors = fit(X_train, train_labels, labels)

# Predict on the test set
pred = predict(n_label_items, vocab, word_counts, log_label_priors, labels, test_lines)

# Print accuracy
print("Accuracy of prediction on test set:", accuracy_score(test_labels, pred))

# Function to test new samples
def test_sample(sample):
    # Preprocess the sample
    sample = remove_tags(sample)
    sample = ' '.join([word for word in sample.split() if word not in stop_words])
    sample = lemmatize_text(sample)
    
    # Vectorize the sample
    sample_vec = vec.transform([sample]).toarray()
    
    # Predict the label
    sample_pred = predict(n_label_items, vocab, word_counts, log_label_priors, labels, [sample])
    label = encoder.inverse_transform(sample_pred)
    return label[0]

input_csv_path = 'recognized_lines.csv'
output_csv_path = 'lines_with_sentiments.csv'

# Clear the output CSV file if it exists
if os.path.exists(output_csv_path):
    os.remove(output_csv_path)

# Read and process lines from the input CSV file
with open(input_csv_path, 'r') as input_csv_file:
    reader = csv.reader(input_csv_file)
    with open(output_csv_path, 'a', newline='') as output_csv_file:
        writer = csv.writer(output_csv_file)
        writer.writerow(['line', 'sentiment'])  # Write header
        for row in reader:
            line = row[0]
            sentiment = test_sample(line)
            writer.writerow([line, sentiment])

print(f"Lines with sentiments have been saved to {output_csv_path}")
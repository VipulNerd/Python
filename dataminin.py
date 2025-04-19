import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_and_preprocess_images(directory, label):
    images = []
    labels = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            # Resize image to a standard size
            img = img.resize((128, 128))
            # Convert to grayscale and normalize
            img_array = np.array(img.convert('L')) / 255.0
            # Flatten the image
            img_array = img_array.flatten()
            images.append(img_array)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Load malignant and benign images
malignant_dir = '/Users/vamhaze/Downloads/Malignant'
benign_dir = '/Users/vamhaze/Downloads/Benign'

X_malignant, y_malignant = load_and_preprocess_images(malignant_dir, 1)
X_benign, y_benign = load_and_preprocess_images(benign_dir, 0)

# Combine the datasets
X = np.vstack((X_malignant, X_benign))
y = np.concatenate((y_malignant, y_benign))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Naïve Bayes Classification
print("\nNaïve Bayes Classification:")
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_scaled, y_train)
y_pred_nb = nb_classifier.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

# 2. Support Vector Machine (SVM) Classification
print("\nSVM Classification:")
svm_classifier = SVC(kernel='rbf', random_state=42)
svm_classifier.fit(X_train_scaled, y_train)
y_pred_svm = svm_classifier.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm)) 
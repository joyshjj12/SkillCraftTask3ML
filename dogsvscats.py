''' NEURAL NETWORKS '''

import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Function to load images from folder based on CSV labels
def load_images_from_csv(csv_file, train_folder, image_size=(64, 64)):
    try:
        data = pd.read_csv(csv_file)
        print(f"Loaded CSV with {len(data)} entries.")
    except Exception as e:
        raise ValueError(f"Error loading CSV file {csv_file}: {e}")
    
    images = []
    labels = []
    
    for index, row in data.iterrows():
        img_id = row['id'] - 1  # Adjust ID to match with image filenames (IDs start from 1)
        label = row['label']
        if label == 0:
            img_filename = f"cat.{img_id}.jpg"
        elif label == 1:
            img_filename = f"dog.{img_id}.jpg"
        else:
            continue  # Skip if label is neither 0 nor 1
        
        img_path = os.path.join(train_folder, img_filename)
        
        try:
            print(f"Loading image: {img_path}")
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(label)
            else:
                print(f"Unable to read image: {img_path}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    return images, labels

# Example usage:
train_folder = r"C:\Users\joysh\task3\train\train"
csv_file = r"C:\Users\joysh\task3\sampleSubmission.csv"

try:
    train_images, train_labels = load_images_from_csv(csv_file, train_folder)

    if len(train_images) == 0:
        raise ValueError("No images found in the dataset.")

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Normalize the pixel values
    train_images = train_images / 255.0

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

    # Define your model architecture (example CNN)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Output layer for binary classification (cat vs dog)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {accuracy}")

    # Save the model if needed
    # model.save('cat_dog_classifier.h5')

except ValueError as ve:
    print(f"Error: {ve}")

''' SVM  '''


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load and preprocess a single image
def load_image(args):
    img_path, label = args
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.flatten()
        return img, label
    else:
        return None, None

# Function to load images in batches
def load_images_in_batches(base_path, batch_size=1000):
    X = []
    y = []
    total_images = 25000
    for i in tqdm(range(0, total_images, batch_size), desc="Loading images in batches"):
        batch = []
        for j in range(i, min(i + batch_size, total_images)):
            if j < 12500:
                img_path = os.path.join(base_path, f"cat.{j}.jpg")
                label = 0
            else:
                img_path = os.path.join(base_path, f"dog.{j-12500}.jpg")
                label = 1
            batch.append((img_path, label))
        
        with Pool(cpu_count()) as pool:
            results = pool.map(load_image, batch)
        
        for img, label in results:
            if img is not None:
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Path to images directory
    base_path = "C:\\Users\\joysh\\task3\\train\\train"

    # Load images and labels
    logging.info("Starting to load images...")
    X, y = load_images_in_batches(base_path)
    logging.info("Finished loading images.")

    # Split data into training and validation sets
    logging.info("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize SVM model
    svm_model = SVC(kernel='linear', C=1.0, gamma='auto')

    # Train SVM model
    logging.info("Starting to train SVM model...")
    svm_model.fit(X_train, y_train)
    logging.info("Finished training SVM model.")

    # Predict on validation set
    y_pred = svm_model.predict(X_val)

    # Evaluate model
    accuracy = accuracy_score(y_val, y_pred)
    logging.info(f"Validation accuracy: {accuracy:.4f}")

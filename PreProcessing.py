import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset folders
DATA_DIRS = ["Acne_Data.csv", "Lupus_Data.csv", "Rashes_Data.csv", "./Dataset4"]  # Update paths as needed

# Verify if dataset folders exist
for data_dir in DATA_DIRS:
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Error: Dataset folder '{data_dir}' not found! Please check the path.")
    else:
        print(f"Dataset folder found at {data_dir}")

# Define constants
IMG_SIZE = (128, 128)  # Resize images to 128x128


# Function to load CSV file from GitHub
def load_csv_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        return df
    else:
        raise Exception("Error: Unable to fetch CSV file from GitHub.")


# Function to load and preprocess images from multiple folders
def load_images_from_folders(data_dirs, img_size):
    X, y, class_names = [], [], {}
    label_index = 0

    for data_dir in data_dirs:
        labels = os.listdir(data_dir)  # Get folder names as labels
        for label in labels:
            class_dir = os.path.join(data_dir, label)
            if not os.path.isdir(class_dir):
                continue

            if label not in class_names:
                class_names[label] = label_index
                label_index += 1

            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)  # Resize to target size
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
                X.append(img)
                y.append(class_names[label])  # Assign numeric label

    return np.array(X), np.array(y), class_names


# Load dataset
X, y, class_mapping = load_images_from_folders(DATA_DIRS, IMG_SIZE)
X = X / 255.0  # Normalize images

# Ensure there are enough samples for splitting
if X.shape[0] < 2:
    raise ValueError("Dataset is too small to split. Need at least 2 samples for a valid split.")

# Split dataset into training (80%) and temp (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if X_temp has enough samples for splitting further
if X_temp.shape[0] < 2:
    print(f"Not enough samples in X_temp ({X_temp.shape[0]}). Skipping further split.")
    X_val, X_test = X_temp, X_temp
    y_val, y_test = y_temp, y_temp
else:
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print dataset sizes
print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
print(f"Validation set: {X_val.shape}, Labels: {y_val.shape}")
print(f"Test set: {X_test.shape}, Labels: {y_test.shape}")

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

# Display a random image from the dataset
plt.imshow(X_train[0])
plt.title("Sample Preprocessed Image")
plt.axis("off")
plt.show()

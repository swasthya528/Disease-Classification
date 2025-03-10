import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Verify if dataset folder exists in Google Drive
DATA_DIR = "/content/drive/MyDrive/Dataset"  # Correct dataset path

# Ensure the dataset folder exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Error: Dataset folder '{DATA_DIR}' not found! Please check the path.")
else:
    print(f"Dataset folder found at {DATA_DIR}")

# Define constants
IMG_SIZE = (128, 128)  # Resize images to 128x128

# Function to load and preprocess images
def load_images(data_dir, img_size):
    labels = os.listdir(data_dir)  # Get folder names as labels
    X, y = [], []

    for idx, label in enumerate(labels):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, img_size)  # Resize to target size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
            X.append(img)
            y.append(idx)  # Assign numeric label

    return np.array(X), np.array(y), labels

# Load dataset
X, y, class_names = load_images(DATA_DIR, IMG_SIZE)

# Normalize images
X = X / 255.0

# Ensure there are enough samples for splitting
if X.shape[0] < 2:
    raise ValueError("Dataset is too small to split. Need at least 2 samples for a valid split.")

# Split dataset into training (80%) and temp (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if X_temp has enough samples for splitting further
if X_temp.shape[0] < 2:
    print(f"Not enough samples in X_temp ({X_temp.shape[0]}). Skipping further split.")
    # In this case, use all data for training and testing
    X_val, X_test = X_temp, X_temp
    y_val, y_test = y_temp, y_temp
else:
    # Now split the 20% into validation and test sets (50%/50%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print dataset sizes
print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
print(f"Validation set: {X_val.shape}, Labels: {y_val.shape}")
print(f"Test set: {X_test.shape}, Labels: {y_test.shape}")

# Data augmentation (rotation, shifting, flipping)
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

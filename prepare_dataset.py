import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Function to load data
def load_data(data_dir, img_size=(128, 128)):
    X, y = [], []
    class_labels = sorted(os.listdir(data_dir))  # Folder names as class labels
    for label, class_name in enumerate(class_labels):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y), class_labels

# Specify your dataset path
data_dir = r"C:\Users\Adil Khan\Desktop\AI Hackathon\Ai 3\dataset"
X, y, class_labels = load_data(data_dir)

# Normalize images and one-hot encode labels
X = X / 255.0  # Scale pixel values
y = tf.keras.utils.to_categorical(y, len(class_labels))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
np.save('class_labels.npy', class_labels)

print("Dataset preparation completed.")

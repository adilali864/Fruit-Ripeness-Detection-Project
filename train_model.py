import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import os

# Create output directory if it doesn't exist
os.makedirs('model_outputs', exist_ok=True)

# Set UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Load and preprocess data
X_train = np.load('processed_dataset/X_train.npy')
X_test = np.load('processed_dataset/X_test.npy')
y_train = np.load('processed_dataset/y_fruit_train.npy')
y_test = np.load('processed_dataset/y_fruit_test.npy')

# Normalize pixel values 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Custom learning rate scheduler
def lr_schedule(epoch):
    initial_lr = 0.001
    drop_rate = 0.5
    epochs_drop = 10.0
    lr = initial_lr * (drop_rate ** np.floor((1 + epoch) / epochs_drop))
    return max(lr, 0.00001)  # Minimum learning rate

# Advanced data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Improved model architecture with input layer
model = Sequential([
    # Input layer
    tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
    
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Dense Layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(y_train[0]), activation='softmax')
])

# Compile with Adam optimizer and fixed learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy', 
        tf.keras.metrics.Precision(name='precision'), 
        tf.keras.metrics.Recall(name='recall')
    ]
)

# Enhanced callbacks
callbacks = [
    # Learning Rate Scheduler
    LearningRateScheduler(lr_schedule),
    
    # Early Stopping
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce Learning Rate
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    ),
    
    # Model Checkpoint
    ModelCheckpoint(
        filepath='model_outputs/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
]

# Train the model with data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=100,
    steps_per_epoch=len(X_train) // 32,
    callbacks=callbacks,
    verbose=1
)

# Save the final model
model.save('model_outputs/final_model.keras')

# Print detailed training results
print("\nModel Training Completed!")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
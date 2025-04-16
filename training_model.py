import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Dataset paths
DATA_DIR = r"C:\Users\PC\Desktop\project\Crop_Disease_Predictor\plantvillage dataset\color"
MODEL_PATH = "crop_model.keras"

# Fixed 38 classes
CLASS_NAMES = sorted([
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
])

# Data generators (lighter augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,  # Reduced for stability
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_dataset = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(64, 64),
    batch_size=32,
    class_mode='sparse',
    subset='training',
    seed=42
)

val_dataset = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(64, 64),
    batch_size=32,
    class_mode='sparse',
    subset='validation',
    seed=42
)

# Verify class count
logger.info(f"Found {len(train_dataset.class_indices)} classes in training data")
if len(train_dataset.class_indices) != 38:
    raise ValueError(f"Expected 38 classes, but found {len(train_dataset.class_indices)}")

# Model (your 79% CNN)
def create_model():
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),  # Added for capacity
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),  # Increased capacity
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(38, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Adjusted lr
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train
logger.info("Starting training...")
model = create_model()
history = model.fit(
    train_dataset,
    epochs=6,  # Shortened for speed
    validation_data=val_dataset,
    verbose=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)]
)

# Save
model.save(MODEL_PATH)
logger.info(f"Model saved to {MODEL_PATH}")
logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
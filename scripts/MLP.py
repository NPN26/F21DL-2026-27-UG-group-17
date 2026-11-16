'''
Custom MLP for BloodMNIST (8-class classification)
Architecture: Input -> Flatten -> Dense(512) -> Dense(256) -> Dense(128) -> Output(8)
'''

# Imports
import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Data preprocessing
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score

# Model
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Seed
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Dataset path
PATH = "data/raw/bloodmnist_128.npz"

df_raw = np.load(PATH)

# Input resolution
# 32 x 32 x 3 = 3,072 flattened features (matches the architecture spec).
# Set to 128 to use full resolution -> 49,152 features (much larger model).
IMG_SIZE = 32

# Assigning the data and labels
TRAIN = df_raw['train_images']
VAL = df_raw['val_images']
TEST = df_raw['test_images']

train_labels = df_raw['train_labels'].ravel()
val_labels = df_raw['val_labels'].ravel()
test_labels = df_raw['test_labels'].ravel()

# Resize once, up front (ImageDataGenerator.flow does not resize)
TRAIN = tf.image.resize(TRAIN, (IMG_SIZE, IMG_SIZE)).numpy()
VAL = tf.image.resize(VAL, (IMG_SIZE, IMG_SIZE)).numpy()
TEST = tf.image.resize(TEST, (IMG_SIZE, IMG_SIZE)).numpy()

CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "immature granulocyte",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]

# Class weights for handling class imbalance
classes = np.unique(train_labels)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_labels
)
class_weights = dict(zip(classes, weights))

# Split preps
# No pretrained backbone here, so scale pixels to [0, 1] instead of
# using an application-specific preprocess_input.
prep_dataTrain = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10
)

prep_val = ImageDataGenerator(rescale=1.0 / 255)
prep_test = ImageDataGenerator(rescale=1.0 / 255)

# Data generators
train_data = prep_dataTrain.flow(
    TRAIN,
    train_labels,
    batch_size=32,
    shuffle=True,
    seed=SEED
)

val_data = prep_val.flow(
    VAL,
    val_labels,
    batch_size=32,
    shuffle=False
)

test_data = prep_test.flow(
    TEST,
    test_labels,
    batch_size=32,
    shuffle=False
)

# Model Definition
model = Sequential([
    layers.Input(shape=(128, 128, 3)),

    # Reduce spatial dimensions before flattening
    layers.Resizing(32, 32),

    layers.Flatten(),

    # Hidden layer 1
    layers.Dense(512, use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.4),

    # Hidden layer 2
    layers.Dense(256, use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.3),

    # Hidden layer 3
    layers.Dense(128, use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.2),

    # Classification layer
    layers.Dense(len(CLASS_NAMES), activation='softmax')
], name="BloodMLP")

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Early stopping and model checkpointing
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'models/MLP_custom.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    class_weight=class_weights,
    callbacks=[early_stopping, checkpoint]
)

# Save training history for report figures
pd.DataFrame(history.history).to_csv('results/MLP_custom_history.csv', index=False)

# Final evaluation on the held-out test set
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

y_pred = np.argmax(model.predict(test_data), axis=1)

print(classification_report(test_labels, y_pred, target_names=CLASS_NAMES))

macro_f1 = f1_score(test_labels, y_pred, average='macro')
print(f"Macro F1: {macro_f1:.4f}")
# Imports
import numpy as np
import tensorflow as tf

# Data preprocessing
from sklearn.utils.class_weight import compute_class_weight

# Model
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Seed
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Dataset path
PATH = "data/raw/bloodmnist_128.npz"

df_raw = np.load(PATH)

# Assigning the data and labels
TRAIN = df_raw['train_images']
VAL = df_raw['val_images']
TEST = df_raw['test_images']

train_labels = df_raw['train_labels'].ravel()
val_labels = df_raw['val_labels'].ravel()
test_labels = df_raw['test_labels'].ravel()

CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "immature granulocyte",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]

# Class weights for handling class imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights))

# Split preps
prep_dataTrain = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    seed=SEED
)

prep_val = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

prep_test = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

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
    shuffle=False,
)

test_data = prep_test.flow(
    TEST,
    test_labels,
    batch_size=32,
    shuffle=False,
)

# Model Definition
base_model = EfficientNetB5(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

# Freeze the base model
base_model.trainable = False

# Add custom layers on top of the base model
output = base_model.output
pooling = GlobalAveragePooling2D()(output)
neurons = Dense(256, activation='relu')(pooling)
dropout = Dropout(0.5)(neurons)
predictions = Dense(len(CLASS_NAMES), activation='softmax')(dropout)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Early stopping and model checkpointing
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'models/EfficentNetB5_transfer_learning.keras',
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
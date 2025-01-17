import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Paths to dataset in Google Drive
plain_path = '/content/drive/My Drive/dataset/plain'
sunglasses_path = '/content/drive/My Drive/dataset/sunglasses'

# Create temporary directories for train and validation splits
base_dir = '/content/temp_dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

plain_train_dir = os.path.join(train_dir, 'plain')
plain_val_dir = os.path.join(val_dir, 'plain')
sunglasses_train_dir = os.path.join(train_dir, 'sunglasses')
sunglasses_val_dir = os.path.join(val_dir, 'sunglasses')

os.makedirs(plain_train_dir, exist_ok=True)
os.makedirs(plain_val_dir, exist_ok=True)
os.makedirs(sunglasses_train_dir, exist_ok=True)
os.makedirs(sunglasses_val_dir, exist_ok=True)

def split_data(source_dir, train_dir, val_dir, split_ratio=0.2):
    files = os.listdir(source_dir)
    random.shuffle(files)
    split_point = int(len(files) * split_ratio)
    val_files = files[:split_point]
    train_files = files[split_point:]
    
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))
    
    for file in val_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, file))

# Split data
split_data(plain_path, plain_train_dir, plain_val_dir)
split_data(sunglasses_path, sunglasses_train_dir, sunglasses_val_dir)

# Data Preparation with Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

val_datagen = ImageDataGenerator(rescale=1./255)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Model Definition using Transfer Learning (VGG16)
base_model = tf.keras.applications.VGG16(input_shape=(64, 64, 3),
                                         include_top=False,
                                         weights='imagenet')

base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for learning rate scheduling and early stopping
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Model Training
history = model.fit(
    train_gen,
    epochs=100,
    validation_data=val_gen,
    callbacks=[reduce_lr, early_stopping]
)

# Save the model
model.save('/content/drive/My Drive/sunglasses_detector_model.h5')

# Clean up temporary directories
shutil.rmtree(base_dir)

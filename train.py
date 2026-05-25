# train.py

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# SETTINGS


IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 5

TRAIN_PATH = r"C:\Users\Home\Downloads\archive.zip\Dataset\train"
TEST_PATH = r"C:\Users\Home\Downloads\archive.zip\Dataset\test"

# DATA GENERATORS


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# MODEL


base_model = MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,
    weights='imagenet',
    alpha=0.35
)

# Freeze pretrained layers
base_model.trainable = False

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dropout(0.3),

    Dense(1, activation='sigmoid')
])


# COMPILE MODEL


model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# CALLBACKS


early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=2,
    restore_best_weights=True
)

# TRAIN MODEL


print("\nTraining Started...\n")

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# TEST MODEL


print("\nTesting Model...\n")

train_loss, train_acc = model.evaluate(train_data)

test_loss, test_acc = model.evaluate(test_data)

print(f"\nTrain Accuracy: {train_acc*100:.2f}%")

print(f"Test Accuracy : {test_acc*100:.2f}%")


# SAVE MODEL

model.save("deepfake_model.keras")

print("\nModel Saved Successfully!")
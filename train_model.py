import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import json

DATASET_PATH = "dataset"
MODEL_PATH = "model/crop_disease_model.keras"
CLASS_PATH = "model/classes.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# LOAD DATASET
# -----------------------------

print("Loading PlantVillage dataset...")
print("This may take some time...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
number_of_classes = len(class_names)

print("\n================================")
print("NUMBER OF CLASSES:", number_of_classes)
print("================================")

for i, name in enumerate(class_names):
    print(i, ":", name)

# Save classes
os.makedirs("model", exist_ok=True)

with open(CLASS_PATH, "w") as f:
    json.dump(class_names, f)

print("\nClass names saved.")

# -----------------------------
# PERFORMANCE
# -----------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# -----------------------------
# DATA AUGMENTATION
# -----------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1)
])

# -----------------------------
# MOBILE NET V2
# -----------------------------

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# First freeze base model
base_model.trainable = False

# -----------------------------
# BUILD MODEL
# -----------------------------

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(256, activation="relu")(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    number_of_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# -----------------------------
# STAGE 1
# -----------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\n================================")
print("STAGE 1 TRAINING")
print("================================")

callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=0.00001
    )
]

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=8,
    callbacks=callbacks
)

# -----------------------------
# STAGE 2 - FINE TUNING
# -----------------------------

print("\n================================")
print("STAGE 2 - FINE TUNING")
print("================================")

base_model.trainable = True

# Freeze early layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    callbacks=callbacks
)

# -----------------------------
# FINAL EVALUATION
# -----------------------------

print("\n================================")
print("FINAL MODEL EVALUATION")
print("================================")

loss, accuracy = model.evaluate(validation_dataset)

print("Validation Loss:", round(float(loss), 4))

print(
    "Validation Accuracy:",
    round(float(accuracy) * 100, 2),
    "%"
)

# -----------------------------
# SAVE MODEL
# -----------------------------

model.save(MODEL_PATH)

print("\n================================")
print("TRAINING COMPLETE!")
print("================================")

print("\nModel saved at:")
print(MODEL_PATH)

print("\nClasses saved at:")
print(CLASS_PATH)
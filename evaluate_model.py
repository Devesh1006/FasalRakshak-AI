import tensorflow as tf
import numpy as np
import json

DATASET_PATH = "dataset"
MODEL_PATH = "model/crop_disease_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = validation_dataset.class_names

print("\nNumber of classes:", len(class_names))

print("\nEvaluating model...")
loss, accuracy = model.evaluate(validation_dataset)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("Validation Loss:", round(float(loss), 4))
print("Validation Accuracy:", round(float(accuracy) * 100, 2), "%")

print("==============================")

# Check several individual predictions
print("\nChecking sample predictions...\n")

for images, labels in validation_dataset.take(1):

    predictions = model.predict(images, verbose=0)

    for i in range(min(10, len(images))):

        predicted_index = np.argmax(predictions[i])
        actual_index = labels[i].numpy()

        confidence = predictions[i][predicted_index] * 100

        print("Actual   :", class_names[actual_index])
        print("Predicted:", class_names[predicted_index])
        print("Confidence:", round(float(confidence), 2), "%")
        print("--------------------------------")
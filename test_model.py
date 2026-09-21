import tensorflow as tf
import numpy as np
import json
import os
import random
from PIL import Image

MODEL_PATH = "model/crop_disease_model.keras"
DATASET_PATH = "dataset"
CLASS_PATH = "model/classes.json"

IMAGE_SIZE = (224, 224)

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

print("Model loaded!")
print("Number of classes:", len(class_names))

print("\nTesting multiple images...\n")

test_classes = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Apple___Apple_scab"
]

for class_name in test_classes:

    folder = os.path.join(DATASET_PATH, class_name)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(files)

    files = files[:10]

    correct = 0

    print("========================================")
    print("EXPECTED:", class_name)
    print("========================================")

    for file in files:

        image_path = os.path.join(folder, file)

        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        

        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        

        prediction = model.predict(img_array, verbose=0)

        predicted_index = np.argmax(prediction[0])

        predicted_class = class_names[predicted_index]

        confidence = prediction[0][predicted_index] * 100

        if predicted_class == class_name:
            correct += 1
            result = "CORRECT"
        else:
            result = "WRONG"

        print(
            result,
            "| Predicted:",
            predicted_class,
            "| Confidence:",
            round(float(confidence), 2),
            "%"
        )

    accuracy = (correct / len(files)) * 100

    print()
    print("Correct:", correct, "/", len(files))
    print("Accuracy:", round(accuracy, 2), "%")
    print()


print("========================================")
print("TEST COMPLETE")
print("========================================")
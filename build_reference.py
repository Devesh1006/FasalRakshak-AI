import os
import json
import random
import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = "model/crop_disease_model.keras"
CLASS_PATH = "model/classes.json"
DATASET_PATH = "dataset"

OUTPUT_PATH = "model/reference_data.npz"

IMAGE_SIZE = (224, 224)

# Number of images used to build each class reference
REFERENCE_IMAGES_PER_CLASS = 50

# Number of separate images used to calibrate similarity threshold
CALIBRATION_IMAGES_PER_CLASS = 20

BATCH_SIZE = 32

RANDOM_SEED = 123


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

print(
    "Number of classes:",
    len(class_names)
)


# =========================================================
# FIND FEATURE LAYER
# =========================================================

feature_layer = None

for layer in reversed(model.layers):

    if isinstance(layer, tf.keras.layers.Dense):

        if layer.units != len(class_names):

            feature_layer = layer
            break


if feature_layer is None:

    print(
        "ERROR: Could not find feature layer."
    )

    exit()


print(
    "Feature layer:",
    feature_layer.name
)

print(
    "Feature size:",
    feature_layer.units
)


feature_model = tf.keras.Model(
    inputs=model.input,
    outputs=feature_layer.output
)


# =========================================================
# PREPARE IMAGE
# =========================================================

def load_image(path):

    image = Image.open(path).convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    array = np.array(
        image,
        dtype=np.float32
    )

    return array


# =========================================================
# GET FILES
# =========================================================

def get_image_files(class_name):

    folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    files = []

    for file in os.listdir(folder):

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            files.append(
                os.path.join(
                    folder,
                    file
                )
            )

    random.shuffle(files)

    return files


# =========================================================
# EXTRACT FEATURES
# =========================================================

def extract_features(files):

    images = []

    for path in files:

        try:

            images.append(
                load_image(path)
            )

        except Exception:

            print(
                "Skipping:",
                path
            )


    if not images:

        return np.empty(
            (0, feature_layer.units)
        )


    images = np.array(
        images,
        dtype=np.float32
    )


    features = feature_model.predict(
        images,
        batch_size=BATCH_SIZE,
        verbose=0
    )


    # L2 normalize each feature vector
    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True
    )

    norms = np.maximum(
        norms,
        1e-10
    )

    features = (
        features / norms
    )

    return features


# =========================================================
# BUILD CLASS CENTROIDS
# =========================================================

centroids = []

calibration_similarities = []

print("\n========================================")
print("BUILDING CLASS REFERENCES")
print("========================================")


for class_index, class_name in enumerate(
    class_names
):

    print(
        f"\n[{class_index + 1}/{len(class_names)}] "
        f"{class_name}"
    )


    files = get_image_files(
        class_name
    )


    minimum_required = (
        REFERENCE_IMAGES_PER_CLASS
        + CALIBRATION_IMAGES_PER_CLASS
    )


    if len(files) < minimum_required:

        print(
            "WARNING: Not enough images."
        )

        reference_files = files

        calibration_files = []

    else:

        reference_files = files[
            :REFERENCE_IMAGES_PER_CLASS
        ]

        calibration_files = files[
            REFERENCE_IMAGES_PER_CLASS:
            REFERENCE_IMAGES_PER_CLASS
            + CALIBRATION_IMAGES_PER_CLASS
        ]


    # -----------------------------------------------------
    # Reference features
    # -----------------------------------------------------

    reference_features = extract_features(
        reference_files
    )


    if len(reference_features) == 0:

        print(
            "ERROR: No features extracted."
        )

        exit()


    centroid = np.mean(
        reference_features,
        axis=0
    )


    # Normalize centroid
    centroid_norm = np.linalg.norm(
        centroid
    )

    centroid = (
        centroid /
        max(centroid_norm, 1e-10)
    )


    centroids.append(
        centroid
    )


    # -----------------------------------------------------
    # Calibration features
    # -----------------------------------------------------

    calibration_features = extract_features(
        calibration_files
    )


    if len(calibration_features) > 0:

        similarities = np.dot(
            calibration_features,
            centroid
        )

        calibration_similarities.extend(
            similarities.tolist()
        )


    print(
        "Reference images:",
        len(reference_features)
    )

    print(
        "Calibration images:",
        len(calibration_features)
    )


# =========================================================
# CALCULATE THRESHOLD
# =========================================================

calibration_similarities = np.array(
    calibration_similarities,
    dtype=np.float32
)


print("\n========================================")
print("CALIBRATION")
print("========================================")


if len(calibration_similarities) == 0:

    print(
        "ERROR: No calibration data."
    )

    exit()


# Keep 95% of known-class examples above
# the similarity threshold.
similarity_threshold = float(
    np.percentile(
        calibration_similarities,
        5
    )
)


# Safety bounds
similarity_threshold = max(
    similarity_threshold,
    0.50
)

similarity_threshold = min(
    similarity_threshold,
    0.95
)


print(
    "Calibration samples:",
    len(calibration_similarities)
)

print(
    "Mean similarity:",
    round(
        float(
            np.mean(
                calibration_similarities
            )
        ),
        4
    )
)

print(
    "5th percentile:",
    round(
        float(
            np.percentile(
                calibration_similarities,
                5
            )
        ),
        4
    )
)

print(
    "Similarity threshold:",
    round(
        similarity_threshold,
        4
    )
)


# =========================================================
# SAVE REFERENCES
# =========================================================

centroids = np.array(
    centroids,
    dtype=np.float32
)


np.savez_compressed(

    OUTPUT_PATH,

    centroids=centroids,

    similarity_threshold=
        np.float32(
            similarity_threshold
        )
)


print("\n========================================")
print("REFERENCE BUILD COMPLETE")
print("========================================")

print(
    "Saved:",
    OUTPUT_PATH
)
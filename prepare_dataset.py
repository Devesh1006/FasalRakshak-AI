from datasets import load_dataset
import os

print("Downloading PlantVillage dataset...")
print("This may take some time...")

dataset = load_dataset("mohanty/PlantVillage")

print("\nDataset downloaded successfully!")

print("\nDataset information:")
print(dataset)

print("\nClasses:")

# Get the class names
features = dataset["train"].features

print(features)

# Save class names
if "label" in features:
    labels = features["label"].names

    print("\nNumber of classes:", len(labels))

    for i, label in enumerate(labels):
        print(i, ":", label)

    os.makedirs("model", exist_ok=True)

    with open("model/classes.txt", "w") as file:
        for label in labels:
            file.write(label + "\n")

    print("\nClass names saved to model/classes.txt")
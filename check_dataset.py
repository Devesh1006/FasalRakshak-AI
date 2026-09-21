import os

dataset_path = "dataset"

if not os.path.exists(dataset_path):
    print("ERROR: dataset/color folder not found!")
    exit()

classes = os.listdir(dataset_path)

print("Dataset found!")
print("Number of disease classes:", len(classes))

print("\nFirst 10 classes:")

for class_name in classes[:10]:
    folder = os.path.join(dataset_path, class_name)
    images = os.listdir(folder)

    print(class_name, "->", len(images), "images")
import os

# Root dataset folder
dataset_path = "./flowers_split"
subfolders = ["train", "valid", "test"]

for sub in subfolders:
    for class_name in os.listdir(os.path.join(dataset_path, sub)):
        class_path = os.path.join(dataset_path, sub, class_name)
        if not os.path.isdir(class_path):
            continue
        images = os.listdir(class_path)
        for i, img_name in enumerate(images, 1):
            ext = os.path.splitext(img_name)[1]  # keep original extension
            new_name = f"{i}{ext}"
            os.rename(os.path.join(class_path, img_name), os.path.join(class_path, new_name))

print("All images renamed sequentially!")

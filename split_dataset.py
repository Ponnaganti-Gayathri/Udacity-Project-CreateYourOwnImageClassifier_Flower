import os
import random
import shutil

# Original dataset folder
src_folder = "./flowers"
# New split folder
dest_folder = "./flowers_split"
splits = ['train', 'valid', 'test']

# Split ratios
split_ratio = {'train': 0.7, 'valid': 0.15, 'test': 0.15}

os.makedirs(dest_folder, exist_ok=True)

for class_name in os.listdir(src_folder):
    class_path = os.path.join(src_folder, class_name)
    if not os.path.isdir(class_path):
        continue
    images = os.listdir(class_path)
    random.shuffle(images)
    n_total = len(images)
    n_train = int(split_ratio['train'] * n_total)
    n_valid = int(split_ratio['valid'] * n_total)
    
    split_paths = {
        'train': images[:n_train],
        'valid': images[n_train:n_train+n_valid],
        'test': images[n_train+n_valid:]
    }

    for split in splits:
        split_class_path = os.path.join(dest_folder, split, class_name)
        os.makedirs(split_class_path, exist_ok=True)
        for img in split_paths[split]:
            shutil.copy(os.path.join(class_path, img), split_class_path)

print("Dataset split completed!")

import warnings
warnings.filterwarnings("ignore")
import torch
from torchvision import models
from PIL import Image
from torch import nn
import os
import sys
import random

def process_image(image_path):
    from torchvision import transforms
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return preprocess(img).unsqueeze(0)

def predict(image_path, checkpoint_path, top_k=3):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    arch = checkpoint['arch']
    num_classes = len(checkpoint['class_to_idx'])

    # Build model
    if arch == 'alexnet':
        model = models.alexnet(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(9216, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )
    else:  # vgg16
        model = models.vgg16(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(25088, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    # Load checkpoint weights
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    # Process image
    img_tensor = process_image(image_path)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.exp(output)
        top_probs, top_indices = probs.topk(top_k)
        top_probs = top_probs.squeeze().numpy()
        top_indices = top_indices.squeeze().numpy()

    # Map indices back to class names
    idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
    classes = [idx_to_class[i] for i in top_indices]

    # Print top predictions
    print(f"\nPredictions for image: {os.path.basename(image_path)}\n")
    for i, (cls, prob) in enumerate(zip(classes, top_probs), 1):
        print(f"{i}: {cls} with probability {prob:.4f}")


if __name__ == "__main__":
    checkpoint_path = "checkpoint.pth"

    # Ask user for flower folder
    flower_name = input("Enter flower class folder (e.g., daisy, rose): ").strip()
    folder_path = os.path.join("flowers_split", "test", flower_name)

    # Check if flower folder exists
    if not os.path.exists(folder_path):
        print(f"Flower class '{flower_name}' does not exist in dataset.")
        sys.exit()

    # Pick a random image from folder
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"No images found in folder '{folder_path}'.")
        sys.exit()
    image_name = random.choice(images)
    image_path = os.path.join(folder_path, image_name)

    # Top K is fixed to 3
    top_k = 3

    # Run prediction
    predict(image_path, checkpoint_path, top_k)

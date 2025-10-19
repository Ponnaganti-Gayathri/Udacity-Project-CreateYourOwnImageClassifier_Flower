import warnings
warnings.filterwarnings("ignore")  # suppress warnings

import torch
from torchvision import models
from PIL import Image
import json
import argparse

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

def predict(image_path, checkpoint_path, top_k=3, category_names=None):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    arch = checkpoint['arch']
    if arch == 'alexnet':
        model = models.alexnet(weights=None)
    else:
        model = models.vgg16(weights=None)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Process image
    img_tensor = process_image(image_path)

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = probs.topk(top_k)
        top_probs = top_probs.squeeze().numpy()
        top_indices = top_indices.squeeze().numpy()

    # Map classes
    idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
    classes = [idx_to_class[i] for i in top_indices]

    # Map to names if JSON provided
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[c] for c in classes]

    # Print results
    for i, (cls, prob) in enumerate(zip(classes, top_probs), 1):
        print(f"{i}: {cls} with probability {prob:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("checkpoint")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--category_names", type=str, default=None)
    args = parser.parse_args()

    predict(args.image_path, args.checkpoint, args.top_k, args.category_names)

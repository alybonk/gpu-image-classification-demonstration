## classify.py

```python
import torch
from torchvision import models, transforms
from PIL import Image
import argparse
import time

# Load class labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
import requests
labels = requests.get(LABELS_URL).text.strip().split("\n")

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def classify_image(img_path, device):
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model = models.resnet50(pretrained=True).to(device)
    model.eval()

    with torch.no_grad():
        start = time.time()
        output = model(input_tensor)
        end = time.time()

    probs = torch.nn.functional.softmax(output[0], dim=0)
    top5 = torch.topk(probs, 5)

    print(f"\nInference device: {device}")
    print(f"Time taken: {end - start:.4f} seconds\n")
    for i in range(5):
        print(f"{labels[top5.indices[i]]}: {top5.values[i].item()*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Path to image')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classify_image(args.img, device)
```

---
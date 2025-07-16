# gpu-image-classification-demo

## README.md

# Image Classification on GPU with PyTorch

This demo shows how to run image classification using a pretrained ResNet-50 model on GPU with PyTorch. We compare inference speed on CPU vs GPU.

### 🔧 Requirements
```bash
pip install torch torchvision matplotlib
```

If using Google Colab, make sure to select `Runtime > Change runtime type > GPU`

### 🚀 Run the Script
```bash
python classify.py --img images/dog.jpg
```

Or open `notebook_version.ipynb` in Jupyter or Colab.

---
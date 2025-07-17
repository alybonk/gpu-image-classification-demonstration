# gpu-image-classification-demonstration

## README.md

# Run Image Classification on GPU with PyTorch

This is a quick demo that loads a pretrained ResNet-50 model from PyTorch and uses it to classify images. It compares CPU vs GPU inference speed. This is a very lightweight demo for beginners on how to use and run image classification using python3. 
```bash
pip3 install torch torchvision pillow matplotlib
```

If you're using Google Colab, go to `Runtime > Change runtime type > GPU` before running the notebook.

### To Run as a Script
```bash
python3 classify.py --img images/dog.jpg
```

Or open `notebook_version.ipynb` and try it out interactively.

### Use an actual GPU
If you dont have access to a GPU, theres two options for you:
1. Use Google Colab (easiest, free)
2. Use a cloud provider with a GPU (AWS EC2 instance, GCP VM, etc)

# gpu-image-classification-demonstration

## README.md

# Run Image Classification on GPU with PyTorch

This is a quick demo that loads a pretrained ResNet-50 model from PyTorch and uses it to classify images. It compares CPU vs GPU inference speed. This is a very lightweight demo for beginners on how to use and run image classification using python3.  If you want more basic context for setting this up, I've added some notes to the end of this README.md

### Requirements
```bash
pip install torch torchvision pillow matplotlib
```

If you're using Google Colab, go to `Runtime > Change runtime type > GPU` before running the notebook.

### To Run as a Script
```bash
python classify.py --img images/dog.jpg
```

Or open `notebook_version.ipynb` and try it out interactively.

###
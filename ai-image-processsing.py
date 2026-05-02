###############################################################################
#from imageai.Classification import ImageClassification
#import torch
#from torchvision import models, transforms
#from torchvision.models import ResNet50_Weights
#from torchvision.models import MobileNet_V2_Weights
#from torchvision.models import EfficientNet_B0_Weights
#from PIL import Image
#import os
## -----------------------------
## 1. Load model + pretrained weights
## -----------------------------
#weights = [
#    ResNet50_Weights.DEFAULT,
#    MobileNet_V2_Weights.DEFAULT,
#    EfficientNet_B0_Weights.DEFAULT]
#
#model = models.resnet50(weights=weights[0])
#model = models.mobilenet_v2(weights=weights[1])
#model = models.efficientnet_b0(weights=weights[2])
#model.eval()  # inference mode
#
##weights = weights[0]  # using ResNet50_Weights for preprocessing
##weights = weights[1]  # using MobileNet_V2_Weights for preprocessing
#weights = weights[2]  # using EfficientNet_B0_Weights for preprocessing
## -----------------------------
## 2. Preprocessing pipeline
## -----------------------------
#preprocess = weights.transforms()
## -----------------------------
## 3. Load and preprocess image
## -----------------------------
#execution_path = os.getcwd()
#image_path = os.path.join(execution_path, "giraffe.jpg")
#
#img = Image.open(image_path).convert("RGB")
#input_tensor = preprocess(img).unsqueeze(0)  # add batch dimension
## -----------------------------
## 4. Run inference
## -----------------------------
#with torch.no_grad():
#    output = model(input_tensor)
#    probabilities = torch.nn.functional.softmax(output[0], dim=0)
## -----------------------------
## 5. Get top‑5 predictions
## -----------------------------
#top5_prob, top5_catid = torch.topk(probabilities, 5)
#
#categories = weights.meta["categories"]
#
#for i in range(5):
#    label = categories[top5_catid[i]]
#    prob = top5_prob[i].item() * 100
#    print(f"{label}: {prob:.2f}%")
## -----------------------------
## 5-2-2026::TEST "giraffe.jpg"
## -----------------------------
##C:\..\ai-ml-image-processing-script-main\venv\Lib\site-packages\torchvision\models\inception.py:43: FutureWarning: The default weight initialization of inception_v3 will be changed in future releases of torchvision. If you wish to keep the old behavior (which leads to long initialization times due to scipy/scipy#11299), please set init_weights=True.
##  warnings.warn(
##impala: 8.57%
##leopard: 7.24%
##gazelle: 5.44%
##cheetah: 3.65%
##zebra: 3.32%
###############################################################################
# ABOVE: AI SUGGESTED CODE FOR IMAGE CLASSIFICATION USING PYTORCH AND PRETRAINED MODELS.
# Node: The environment was updated to python 3.14 and run with the given output. The original author's code was changed
#   entirely, so see initial commit to reference that author.
# Microsoft CoPilot is used for the source code generation and suggestions in this repository.
# URL: https://copilot.microsoft.com
# Date 5-2-2026
# Note: The code demonstrates how to use pretrained models from the torchvision library to classify an image. It includes loading the model, preprocessing the image, and displaying the top-5 predictions with their probabilities.
# BELOW: CODE SUGGESTION USES TIMM LIBRARY.
# Note: Source repository is located with this command: py -m pip show timm
# -----------------------------
import timm
import torch
from PIL import Image
from torchvision import transforms
# -----------------------------
# 1. Load the model and add configuration
# -----------------------------
model = timm.create_model("efficientnet_b0", pretrained=True)
model.eval()

config = timm.data.resolve_data_config({}, model=model)
transform = timm.data.create_transform(**config)
# -----------------------------
# 2. Load and preprocess image
# -----------------------------
img = Image.open("40628437283_a17f4d3ccb_k.webp").convert("RGB")
x = transform(img).unsqueeze(0)

with torch.no_grad():
    preds = model(x)
    ##Output is a logits tensor
    #print(f"Predictions:", preds)
# -----------------------------
# 3. Apply softmax to get probabilities
# -----------------------------
import torch.nn.functional as F
probs = F.softmax(preds, dim=1)

##Output is a tensor of probabilities for each class
#print(f"Probabilities:", probs)
# -----------------------------
# 4. Get top-k predictions
# -----------------------------
top5_prob, top5_catid = torch.topk(probs, 5)
# -----------------------------
# 5. Convert class IDs to labels (human readable)
# -----------------------------
import json
import urllib.request

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = urllib.request.urlopen(url).read().decode("utf-8").splitlines()
# -----------------------------
# 6. Get top-5 predictions
# -----------------------------
for i in range(5):
    print(f"{labels[top5_catid[0][i]]}: {top5_prob[0][i].item() * 100:.2f}%")
# -----------------------------
# 5-2-2026::TEST(timm) "40628437283_a17f4d3ccb_k.webp"
# -----------------------------
#(venv) PS C:\..\ai-ml-image-processing-script-main> python .\ai-image-processsing.py
#Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
#fountain: 84.96%
#picket fence: 2.37%
#paintbrush: 0.77%
#volcano: 0.68%
#corn: 0.52%

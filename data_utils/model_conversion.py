import torch
import torch.hub
import torch.onnx
import torchvision
from torchvision import models


onnx_model_path = "./weight_files"

# model = models.
model = torch.hub.load("./weight_files/", "yolov5s", pretrained=False, source="local", verbose=True)
# model = torch.load('./weight_files/best.pt', map_location=torch.device('cpu'))['model']
# model = torch.load('./weight_files/best.pt')

model.eval()

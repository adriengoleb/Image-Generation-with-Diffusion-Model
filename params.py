import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 32
T =  700
model_name = 'model.pth'
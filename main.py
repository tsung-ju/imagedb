import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import os

BATCH_SIZE = 512

mps_device = torch.device("mps")

# DINOv2
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(mps_device)

# Scale down the image to 224x224
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the images from './images' using the transform and ImageFolder
dataset = ImageFolder('./images', transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

image_clstoken = []
image_patchtokens = []
image_regtokens = []

for i, (batch, _) in enumerate(tqdm(loader)):
    if i == 100:
        break
    with torch.inference_mode():
        batch = batch.to(mps_device)
        features = model.forward_features(batch)
    image_clstoken.append(features["x_norm_clstoken"].to("cpu"))
    image_patchtokens.append(features["x_norm_patchtokens"].to("cpu"))
    image_regtokens.append(features["x_norm_regtokens"].to("cpu"))
    
image_clstoken = torch.cat(image_clstoken)
image_patchtokens = torch.cat(image_patchtokens)
image_regtokens = torch.cat(image_regtokens)

torch.save([path for path, _ in dataset.imgs], "image_paths.pt")
torch.save(image_clstoken, "image_clstoken.pt")
torch.save(image_patchtokens, "image_patchtokens.pt")
torch.save(image_regtokens, "image_regtokens.pt")

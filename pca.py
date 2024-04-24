import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from joblib import dump, load

LIMIT = 20000

#image_paths = torch.load("image_paths.pt")
A = torch.load("image_patchtokens.pt")[:LIMIT]
#image_clstoken = torch.load("image_clstoken.pt").numpy()
#image_regtokens = torch.load("image_regtokens.pt").numpy()

A = A.reshape(-1, A.shape[-1])
print("computing mean")
mean = torch.mean(A, dim=0)
print("centering")
A -= mean
print("computing svd")
U, S, Vt = torch.linalg.svd(A, full_matrices=False)

torch.save(Vt , "pca_components.pt")
torch.save(mean, "pca_mean.pt")

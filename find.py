import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


# DINOv2
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

LIMIT = 20000
n_images = 15

# Scale down the image to 224x224
transform = transforms.Compose([
    transforms.Resize(256),
    #transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_pil = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

@torch.jit.script
def sinkhorn(similarities: torch.Tensor, a: torch.Tensor, b: torch.Tensor, reg: float, scale: float, n_iters: int) -> torch.Tensor:
    a = F.pad(a * scale, (0, 1), value=1.)
    b = F.pad(b, (0, 1), value=scale)
    K = F.pad(similarities, (0, 1, 0, 1), value=0.).sub(1).div(reg).exp()
    u = torch.ones(K.shape[0], K.shape[1]) / K.shape[1]
    v = torch.ones(K.shape[0], K.shape[2]) / K.shape[2]
    for it in range(n_iters):
        print(f"Sinkhorn-Knopp iteration {it}")
        u = a / torch.einsum("bij,bj->bi", K, v)
        v = b / torch.einsum("bij,bi->bj", K, u)
    print("computing final scores")
    return torch.einsum("bij,bij,bi,bj->b", K[:, :-1, :-1], similarities, u[:, :-1], v[:, :-1])

with torch.inference_mode():
    # Load and embed the query image
    query_image = Image.open(sys.argv[1]).convert("RGB")
    query_image = transform(query_image).unsqueeze(0)
    query_features = model.forward_features(query_image)
    query_image_patchtokens = query_features["x_norm_patchtokens"]
    query_image_clstoken = query_features["x_norm_clstoken"]
    query_image_regtokens = query_features["x_norm_regtokens"]

    image_paths = torch.load("image_paths.pt", mmap=True)[:LIMIT]
    image_patchtokens = torch.load("image_patchtokens.pt", mmap=True)[:LIMIT]
    image_clstoken = torch.load("image_clstoken.pt", mmap=True)[:LIMIT]
    image_regtokens = torch.load("image_regtokens.pt", mmap=True)[:LIMIT]

    pca_mean = torch.load("pca_mean.pt")
    pca_components = torch.load("pca_components.pt")

    if 1:
        print("normalizing patches")
        image_patchtokens -= pca_mean
        query_image_patchtokens -= pca_mean
        image_patchtokens = F.normalize(image_patchtokens, dim=-1)
        query_image_patchtokens = F.normalize(query_image_patchtokens, dim=-1)
        print("computing foreground masks")
        query_image_foreground = (query_image_patchtokens @ pca_components[:2].T > 0).any(dim=2)
        image_foreground = (image_patchtokens @ pca_components[:2].T > 0).any(dim=2)
        #print("zeroing out background patches")
        #query_image_patchtokens[~query_image_foreground] = 0
        #image_patchtokens[~image_foreground] = 0
        print("computing patch similarities")
        similarities = image_patchtokens @ query_image_patchtokens.squeeze().T

        #print("compute global similarity by directed mean Hausdorff")
        #similarities.clamp_(min=0) # ignore negative similarities
        #scores = similarities.max(dim=2)[0].sum(dim=1)
        #scores /= image_foreground.sum(dim=1).clamp_(min=1) # normalize by the number of foreground patches
        #top_k = torch.topk(scores, n_images)

        print("computing overall similarity by Sinkhorn-Knopp")

        # Transport only the foreground patches
        a = image_foreground.float().div_(image_foreground.sum(dim=1, keepdim=True))
        b = query_image_foreground.float().div_(query_image_foreground.sum(dim=1, keepdim=True))

        reg = .05
        scale = 2.
        n_iters = 20

        scores = sinkhorn(similarities, a, b, scale, reg, n_iters)
        top_k = torch.topk(scores, n_images)

        #distances = 1 - similarities
        #assignment_distances = distances.norm(dim=1, p=-2).mean(dim=1) + distances.norm(dim=2, p=-2).mean(dim=1)

        #similarities = (similarities.max(dim=1)[0].mean(dim=1) + similarities.max(dim=2)[0].mean(dim=1)) / 2
        #similarities = similarities.max(dim=2)[0].mean(dim=1)
        #print("computing distances")
        #similarities = similarities.max(dim=1)[0].mean(dim=1)
        #similarities = similarities.norm(dim=(1, 2))
        #assignment_distances = 1 - similarities
    elif 0:
        pca = load("pca.joblib")
        pca_image_patchtokens = pca.transform(normalize(image_patchtokens.reshape(-1, image_patchtokens.shape[-1]))).reshape(image_patchtokens.shape[0], image_patchtokens.shape[1], -1)
        mean_image_patchtokens = image_patchtokens.mean(axis=1, where=pca_image_patchtokens > 0)
        print(pca_image_patchtokens.shape)
        print(mean_image_patchtokens.shape)
        pca_query_image_patchtokens = pca.transform(normalize(query_image_patchtokens.reshape(-1, query_image_patchtokens.shape[-1]))).reshape(query_image_patchtokens.shape[0], query_image_patchtokens.shape[1], -1)
        mean_query_image_patchtokens = query_image_patchtokens.mean(axis=1, where=pca_query_image_patchtokens > 0)
        assignment_distances = cdist(mean_image_patchtokens, mean_query_image_patchtokens, metric="cosine").reshape(-1)
    elif 0:
        # Hausdorff distance
        assignment_distances = np.zeros(image_patchtokens.shape[0])
        for i in range(len(image_patchtokens)):
            assignment_distances[i] = max(directed_hausdorff(image_patchtokens[i], query_image_patchtokens[0])[0], directed_hausdorff(query_image_patchtokens[0], image_patchtokens[i])[0])
    elif 0:
        distances = cdist(image_patchtokens.reshape(-1, image_patchtokens.shape[-1]), query_image_patchtokens[0]).reshape(image_patchtokens.shape[0], image_patchtokens.shape[1], -1)
        print(distances.shape)
        assignment_distances = np.zeros(image_patchtokens.shape[0])
        for i in range(len(image_patchtokens)):
            row, col = linear_sum_assignment(distances[i])
            assignment_distances[i] = distances[i][row, col].mean()
            # Mean Hausdorff distance
            # assignment_distances[i] = hmean(distances[i].min(axis=0)) + hmean(distances[i].min(axis=1))
    else:
        print("Calculating global similarity")
        #image_clsreg = np.concatenate([image_clstoken, image_regtokens.reshape(image_regtokens.shape[0], -1)], axis=1)
        #query_clsreg = np.concatenate([query_image_clstoken, query_image_regtokens.reshape(1, -1)], axis=1)
        similarity = F.normalize(image_clstoken, dim=-1) @ F.normalize(query_image_clstoken[0], dim=-1)
        assignment_distances = 1 - similarity.squeeze()

    # Show the top 15 images all together
    plt.figure(figsize=(n_images // 5 + 1, 5))
    plt.subplot(n_images // 5 + 1, 5, 3)
    pil_image = Image.open(sys.argv[1])
    pil_image = transform_pil(pil_image)
    plt.imshow(pil_image, extent=(0, 1, 0, 1))
    plt.imshow(query_image_foreground.reshape(16, 16), vmin=0, vmax=1, cmap="hot", alpha=0.5, interpolation="nearest", extent=(0, 1, 0, 1))
    plt.axis("off")
    plt.title("query image", fontsize=8)
    for i, (idx, similarity) in enumerate(zip(top_k.indices, top_k.values)):
        plt.subplot(n_images // 5 + 1, 5, i + 1 + 5)
        pil_image = Image.open(image_paths[idx])
        pil_image = transform_pil(pil_image)
        # Creat aplha mask
        # alphas = similarities[idx].max(dim=0)[0].reshape(16, 16)
        # scale it up to 224x224
        # alphas = torch.kron(alphas, torch.ones(14, 14))
        plt.imshow(pil_image, extent=(0, 1, 0, 1))
        similarity_map = similarities[idx].max(dim=0)[0]
        similarity_map *= image_foreground[idx]
        plt.imshow(similarity_map.reshape(16, 16), vmin=0, vmax=1, cmap="hot", alpha=0.5, interpolation="nearest", extent=(0, 1, 0, 1))
        print(image_paths[idx])
        plt.axis("off")
        plt.colorbar()
        plt.title(f"{similarity:.4f}", fontsize=8)
    plt.show()

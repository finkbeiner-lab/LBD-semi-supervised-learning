import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import torch
import os
import timm

model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

transform = transforms.Compose(
    [
        #transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

pca = PCA(n_components=3)
scaler = MinMaxScaler(clip=True)

def load_and_preprocess_image(image_path: str) -> Image.Image:
    with open(image_path, 'rb') as f:
        img = Image.open(f).convert('RGB')
    return img

def create_overlay_image(original, result, alpha=0.3):
    # Resize result to match the original image size
    result_resized = cv2.resize(result, (original.shape[1], original.shape[0]))
    overlay = (alpha * original + (1 - alpha) * result_resized * 255).astype(np.uint8)
    return overlay

def process_images(images: List[Image.Image], background_threshold: float = 0.5, larger_pca_as_fg: bool = False) -> List[np.ndarray]:
    imgs_tensor = torch.stack([transform(img).to(device) for img in images])

    with torch.no_grad():
        intermediate_features = model.forward_intermediates(imgs_tensor, intermediates_only=True)
        features = intermediate_features[-1].permute(0, 2, 3, 1).reshape(-1, 1536).cpu()

    pca_features = scaler.fit_transform(pca.fit_transform(features))

    if larger_pca_as_fg:
        fg_indices = pca_features[:, 0] > background_threshold
    else:
        fg_indices = pca_features[:, 0] < background_threshold

    fg_features = pca.fit_transform(features[fg_indices])

    scaler.fit(fg_features)
    normalized_features = scaler.transform(fg_features)

    # Prepare the result
    result_img = np.zeros((imgs_tensor.size(0) * 196, 3))  # Assuming 14x14 patches
    result_img[fg_indices] = normalized_features

    imgs_tensor = imgs_tensor.cpu()

    transformed_imgs = []
    for i, img in enumerate(imgs_tensor):
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_np = (img_np * 255).astype(np.uint8)
        transformed_imgs.append(img_np)

    results = [result_img.reshape(imgs_tensor.size(0), 14, 14, 3)[i] for i in range(len(images))]

    return results, transformed_imgs, pca_features

image_paths = [
#"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/outputs/preprocessing/output/PD13_Amygdala_C110-115 David Menassa.svs/17994x_32747y.png",
#"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/outputs/preprocessing/output/PD13_Amygdala_C110-115 David Menassa.svs/17994x_21995y.png"
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/outputs/preprocessing/output/11_063_CG_aSyn_x200.svs/10206x_11228y.png",
#"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/outputs/preprocessing/output/11_063_CG_aSyn_x200.svs/10206x_14812y.png"
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WM_images/11_063_CG_aSyn_x200.svs/grey/11_063_CG_aSyn_x200.svs_x_7659_y_11898.png"
]


images = [load_and_preprocess_image(path) for path in image_paths]
print(len(images))
results, transformed_imgs, pca_features = process_images(images, larger_pca_as_fg=False)

num_images = len(transformed_imgs)

fig, axes = plt.subplots(num_images, 3, figsize=(9, 3 * num_images))

for i, (image, result) in enumerate(zip(transformed_imgs, results)):
    overlay = create_overlay_image(image, result)

    # Original image
    axes[i, 0].imshow(image)
    axes[i, 0].set_title(f"Original Image {i+1}")
    axes[i, 0].axis('off')

    # PCA result image
    axes[i, 1].imshow(result)
    axes[i, 1].set_title(f"Foreground-Only PCA for Image {i+1}")
    axes[i, 1].axis('off')

    # Overlay image
    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title(f"Overlay for Image {i+1}")
    axes[i, 2].axis('off')

#fig.suptitle('PCA Visualizations', fontsize=20)

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/outputs/pca_imgs",'my_figure4.png')) 

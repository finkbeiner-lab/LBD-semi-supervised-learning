import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
#import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import torch
import os
import timm
from glob import glob
from torch.utils.data import DataLoader, Dataset
import multiprocessing

model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)

transform = transforms.Compose(
    [
        #transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, img_path  # Return image and its path
    
pca = PCA(n_components=3)
scaler = MinMaxScaler(clip=True)

def load_and_preprocess_image(image_path: str) -> Image.Image:
    with open(image_path, 'rb') as f:
        img = Image.open(f).convert('RGB')
    return img

def process_images(all_features: np.array, crop_names: List, background_threshold: float = 0.5, larger_pca_as_fg: bool = False) -> List[np.ndarray]:
    """
    imgs_tensor = torch.stack([transform(img).to(device) for img in images])
    batches = torch.split(imgs_tensor, batch_size)
    print(len(batches))
    batched_outputs = [] 
    for i, batch in enumerate(batches):
        print(f"Batch {i} shape: {batch.shape}")
    
        with torch.no_grad():
            intermediate_features = model.forward_intermediates(batch, intermediates_only=True)
            features = intermediate_features[-1].permute(0, 2, 3, 1).reshape(-1, 1536).cpu()
            batched_outputs.append(features)

    all_features = np.concatenate(batched_outputs, axis=0)
    """
    pca_features = scaler.fit_transform(pca.fit_transform(all_features))
    print(pca_features.shape)

    """
    if larger_pca_as_fg:
        fg_indices = pca_features[:, 0] > background_threshold
    else:
        fg_indices = pca_features[:, 0] < background_threshold

    fg_features = pca.fit_transform(all_features[fg_indices])
    crop_names_1 = crop_names[fg_indices]
    
    print(fg_features.shape)
    
    scaler.fit(fg_features)
    normalized_features = scaler.transform(fg_features)
    print(normalized_features.shape)
    return normalized_features, crop_names_1
    """
    scaler.fit(pca_features)
    normalized_features = scaler.transform(pca_features)
    
    return normalized_features, crop_names
    


#TILE_DIR = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/gigapath_syn1_crops" # for syn1 crops
#TILE_DIR = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/tiles/output"
TILE_DIR = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/aggregate_embeddings/test_data/DLB-007-2015-Amygdala-C34-45.svs"

image_paths = glob(os.path.join(TILE_DIR,"*.png"))
"""
svs_files =  glob(os.path.join(TILE_DIR,"*.svs"))
print(svs_files)
image_paths_all = []
for svs_file in svs_files:
    image_paths = glob(os.path.join(svs_file,"*.png"))
    image_paths_all.extend(image_paths)
    #print(len(image_paths))

print(len(image_paths_all))
"""

batch_size = 16 # Adjust batch size based on available VRAM
num_workers = min(multiprocessing.cpu_count(), 16)  # Use multiple CPUs

#dataset = ImageDataset(image_paths_all, transform)
dataset = ImageDataset(image_paths, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# **Extract Features Using GPU (Optimized)**
feature_list = []
crop_names = []

with torch.no_grad():
    for batch_imgs, batch_paths in dataloader:
        batch_imgs = batch_imgs.to(device)
        #intermediate_features = model.forward_intermediates(batch_imgs, intermediates_only=True)
        features = model(batch_imgs) # Shape: [batch_size, feature_dim]
        #features = features.mean(dim=(2, 3))  # Global Average Pooling
        print(features.shape)
        # Convert to CPU NumPy array
        feature_list.extend(features.cpu().numpy())
        #features = intermediate_features[-1].permute(0, 2, 3, 1).reshape(-1, 1536).cpu()
        #print(features.shape)
        # Convert to CPU NumPy array
        #feature_list.append(features.cpu().numpy())
        crop_names.extend(batch_paths)
        #print(batch_paths.shape)
        
print(len(crop_names))
print(len(feature_list))

# **Save Extracted Features**
#features_np = np.concatenate(feature_list, axis=0)  # Convert list to NumPy array
    
features_np = np.array(feature_list)
crop_names_np = np.array(crop_names)
print(features_np.shape)
#print(len(image_paths_all))
#images = [load_and_preprocess_image(path) for path in image_paths_all]
#pca_features, crop_paths = process_images(features_np,crop_names_np, larger_pca_as_fg=False)
#print(pca_features.shape)
print(crop_names_np.shape)

#np.save('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/'+'antibody_data_pca_features.npy', pca_features) 
#np.save('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/'+'antibody_data_image_paths.npy', crop_paths) 

np.save( "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/aggregate_embeddings/test_data/DLB-007-2015-Amygdala-C34-45.svs/"+'features.npy', features_np) 
np.save( "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/aggregate_embeddings/test_data/DLB-007-2015-Amygdala-C34-45.svs/"+'crop_names.npy', crop_names_np) 



"""
np.save('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/'+'syn1_pca_features.npy', pca_features) 
np.save('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/'+'syn1_image_paths.npy', crop_paths) 
np.save('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/'+'syn1_all_features.npy', features_np) 
np.save('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/'+'syn1_all_crop_paths.npy', crop_names) 

features_np = np.load('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/'+'syn1_all_features.npy',allow_pickle = True ) 
crop_names = np.load('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/'+'syn1_all_crop_paths.npy',allow_pickle = True) 

pca_features, crop_paths = process_images(features_np,crop_names, larger_pca_as_fg=False)
np.save('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/'+'syn1_pca_features_all.npy', features_np) 
np.save('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/'+'syn1_pca_crop_paths_all.npy', crop_names) 
"""



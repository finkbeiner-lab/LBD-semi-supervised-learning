from stain_normalizer import StainNormalizer
import os
import h5py
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
#import cv2
import matplotlib.pyplot as plt
from typing import List
import torch
import os
import timm
from glob import glob
from torch.utils.data import DataLoader, Dataset
import multiprocessing
import pandas as pd 
#from skimage.color import rgb2hed, hed2rgb
import cv2

model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)


transform = transforms.Compose(
    [
        #transforms.ColorJitter(brightness=.3, contrast=0.3 ,saturation=0.3,  hue=.3), # enable for augmenting/color jitter
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        #transforms.CenterCrop(224),
        #transforms.ToTensor(),
        
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        
    ]
)

norm_model_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/acd_model_weights.pth"

normalizer = StainNormalizer(_template_dc_mat=np.array([[ 1.1939337 , -0.54269063, -0.28952357],
       [ 0.35927975,  1.1811069 , -0.83746046],
       [-0.13805144, -0.10795316,  1.139746  ]]),_template_w_mat= [np.array(1.5159065, dtype='float32'), np.array(1.8867766, dtype='float32'), np.array(1)],
       model_weights_path =norm_model_path)


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        #img = Image.open(img_path).convert("RGB")
        #try:
        cv2_img = cv2.imread(img_path)
        img = np.asarray([cv2_img])
        #print(img.shape)
        norm_image =normalizer.transform(img[:,:,:,[2,1,0]])
        #print(norm_image.shape)
        img_tensor = torch.from_numpy(norm_image[0]).permute(2, 0, 1).float() / 255.0  # C, H, W
        img = self.transform(img_tensor)
        return img, img_path  # Return image and its path
         #   if cv2_img is None:
        #except:
        #    pass       

    
def load_and_preprocess_image(image_path: str) -> Image.Image:
    with open(image_path, 'rb') as f:
        img = Image.open(f).convert('RGB')
    return img


def get_coords_from_imgname(svs_file):
    csv_path = os.path.join(svs_file,"dataset.csv")
    csv_df = pd.read_csv(csv_path)
    mapped_image_tiles_coords = {}

    for k, x, y in zip(csv_df["image"], csv_df["tile_x"], csv_df["tile_y"]):
        key = k.split("/")[-1]  # Extract the relevant part of the key
        if key not in mapped_image_tiles_coords:
            mapped_image_tiles_coords[key] = [x, y]
    return mapped_image_tiles_coords

  
def extract_embeddings(svs_file,batch_size,num_workers):
    image_paths = glob(os.path.join(svs_file,"*.png"))
    print(len(image_paths))
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feature_list = []
    crop_names = []
    coords = []

    
    for batch_imgs, batch_paths in dataloader:
        batch_imgs = batch_imgs.to(device)
        with torch.no_grad():
            features = model(batch_imgs) # Shape: [batch_size, feature_dim]
            #print(features.shape)
            # Convert to CPU NumPy array
            feature_list.extend(features.cpu().numpy())
            crop_names.extend(batch_paths)
    return feature_list, crop_names



def save_embeddings(mapped_image_tiles_coords,feature_list, crop_names, h5filename):
    coords =  [mapped_image_tiles_coords[k.split("/")[-1]] for k in crop_names]
    features_np = np.array(feature_list)
    str_dtype = h5py.string_dtype(encoding='utf-8')  # Variable-length UTF-8 strings
    crop_names_np = np.array(crop_names, dtype=str_dtype)  # Convert crop names
    coords_np  = np.array(coords)
    with h5py.File(h5filename, "w") as h5f:
        h5f.create_dataset("features", data=features_np)
        h5f.create_dataset("coords", data=coords_np)  # Optional, for indexing
        h5f.create_dataset("crop_names", data=crop_names_np, dtype=str_dtype)  # Store strings properly
    print(h5filename+" saved!")


def run_for_single_slide_imgs(svs_file, batch_size, num_workers, dir_path):
    mapped_image_tiles_coords = get_coords_from_imgname(svs_file)
    print("mapped_image_tiles_coords", len(mapped_image_tiles_coords))
    feature_list, crop_names = extract_embeddings(svs_file,batch_size,num_workers)
    #img_list, feature_list, crop_names = extract_dab_embeddings(svs_file,batch_size,num_workers)
    print(len(feature_list))
    h5filename = svs_file.split("/")[-1].replace(".svs",".h5")
    h5filename = os.path.join(dir_path,h5filename)
    print(h5filename)
    save_embeddings(mapped_image_tiles_coords,feature_list, crop_names, h5filename)

def tensor_2_img(inp):
    """Imshow for Tensor."""
    inp = inp.squeeze(0)
    print(inp.shape)
    #inp = inp.numpy().transpose((0, 3, 2, 1))
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    print(inp.shape)
    #mean1 = np.array(mean)
    #std1 = np.array(std)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = inp*255
    pil_img = Image.fromarray((inp).astype(np.uint8))
    pil_img.save("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/norm_tiles/test.png")

if __name__ == '__main__':
    path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/tiles/output"
    batch_size = 16 # Adjust batch size based on available VRAM
    num_workers = min(multiprocessing.cpu_count(), 16)  # Use multiple CPUs
    dir_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/norm_tiles"
    svs_folders = glob(os.path.join(path, "*.svs"))
    completed_slides =  glob(os.path.join(dir_path,"*.h5"))
    completed_slides = [x.split("/")[-1] for x in completed_slides]
    completed_slides.append("PDD-PD159-Striatum-C110-115.h5")
    for svs_file in svs_folders:
        if svs_file.split("/")[-1].replace(".svs",".h5") not in completed_slides:
            try:
                run_for_single_slide_imgs(svs_file, batch_size, num_workers, dir_path)
            except pd.errors.EmptyDataError:
                print(svs_file)
        else:
            print("h5 already done")
    
    """
    image_paths = glob(os.path.join(svs_file,"*.png"))
    dataset = ImageDataset(image_paths[:5], transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    #with torch.no_grad():
    for batch_imgs, batch_paths in dataloader:
        batch_imgs = batch_imgs.to(device)
        print(batch_imgs.shape)
        tensor_2_img(batch_imgs[0])
        with torch.no_grad():
            features = model(batch_imgs) # Shape: [batch_size, feature_dim]
            print(features.shape)
    """
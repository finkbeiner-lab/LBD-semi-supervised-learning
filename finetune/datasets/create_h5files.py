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
from skimage.color import rgb2hed, hed2rgb

model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)


transform = transforms.Compose(
    [
        #transforms.ColorJitter(brightness=.3, contrast=0.3 ,saturation=0.3,  hue=.3), # enable for augmenting/color jitter
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        #transforms.CenterCrop(224),
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
    

class DABImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        # Separate the stains from the IHC image
        ihc_hed = rgb2hed(img)
        null = np.zeros_like(ihc_hed[:, :, 0])
        ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
        #print(ihc_d)
        #ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
        #ihc_d = np.clip(ihc_d, 0, 1)
        #plot_img(ihc_d, img_path.split("/")[-1])
        im = Image.fromarray((ihc_d*255).astype(np.uint8), mode="RGB")
        img = self.transform(im)
        return ihc_d, img, img_path  # Return image and its path


    
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

    with torch.no_grad():
        for batch_imgs, batch_paths in dataloader:
            batch_imgs = batch_imgs.to(device)
            features = model(batch_imgs) # Shape: [batch_size, feature_dim]
            #print(features.shape)
            # Convert to CPU NumPy array
            feature_list.extend(features.cpu().numpy())
            crop_names.extend(batch_paths)
    return feature_list, crop_names


def extract_dab_embeddings(svs_file,batch_size,num_workers):
    image_paths = glob(os.path.join(svs_file,"*.png"))
    print(len(image_paths))
    dataset = DABImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feature_list = []
    crop_names = []
    coords = []
    batch_imgs_list = []


    with torch.no_grad():
        for batch_ihc_d, batch_imgs, batch_paths in dataloader:
            batch_imgs_list.extend(batch_ihc_d)
            batch_imgs = batch_imgs.to(device)
            features = model(batch_imgs) # Shape: [batch_size, feature_dim]
            #print(features.shape)
            # Convert to CPU NumPy array
            feature_list.extend(features.cpu().numpy())
            crop_names.extend(batch_paths)
            
    return batch_imgs_list, feature_list, crop_names





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



def read_assets_from_h5(h5_path: str) -> tuple:
        '''Read the assets from the h5 file'''
        assets = {}
        attrs = {}
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                assets[key] = f[key][:]
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
        return assets, attrs
    
#assets, attrs = read_assets_from_h5("tile_embeddings.h5")

#print(assets['coords'].shape)
#print(assets['features'].shape)
#print(assets['coords'])
#print(assets['features'])

#np.load('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/'+'antibody_data_pca_features.npy') 
#np.save('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/'+'antibody_data_image_paths.npy', crop_paths) 
#path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/datasets/data/GigaPath_PANDA_embeddings/h5_files/1da207ca934ce44dc9b3dd8809d0b834.h5"
#svs_file = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/tiles/output/2008-134-C34-45-Amygdala.svs"



if __name__ == '__main__':
    path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/tiles/output"
    batch_size = 16 # Adjust batch size based on available VRAM
    num_workers = min(multiprocessing.cpu_count(), 16)  # Use multiple CPUs
    
    dir_path =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/h5_files_full"
    #dir_path =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/ColorJitter_h5_files_test"
    #dir_path =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/DAB_ColorJitter_h5_files"


    #TILE_DIR = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/tiles"
    
    #tile_one_slide(slide_path, save_dir=tmp_dir, level=1)
    completed_slides =  glob(os.path.join(dir_path,"*.h5"))
    #completed_slides1 =  glob(os.path.join(dir_path,"*/*.h5"))
    #completed_slides.extend(completed_slides1)
    #completed_slides = [x.split("/")[-1] for x in completed_slides]
    svs_folders = glob(os.path.join(path, "*.svs"))
    print(svs_folders)
    for svs_file in svs_folders:
        if svs_file.split("/")[-1].replace(".svs",".h5") not in completed_slides:
            try:
                run_for_single_slide_imgs(svs_file, batch_size, num_workers, dir_path)
            except pd.errors.EmptyDataError:
                print(svs_file)
        else:
            print("h5 already done")
    """

    dir_path =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/ColorJitter_h5_files_test"
    test_csv_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/train_test_split/lbd/test_0.csv"
    tile_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/tiles/output"    
    test_csv = pd.read_csv(test_csv_path)
    for i in range(len(test_csv)):
        slide_id = test_csv["slide_id"].iloc[i]
        svs_file = os.path.join(tile_path, slide_id)
        print(svs_file)
        run_for_single_slide_imgs(svs_file, batch_size, num_workers, dir_path)
    """
    
import os
import cv2
import numpy as np
from glob import glob
import os
import sys
from stain_normalizer import StainNormalizer
from acd import ACDModel
#from stain_normalizer import StainNormalizer
import torch
import random 
import h5py
random.seed(0)

source_image_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/tiles/output"

svs_folders = glob(os.path.join(source_image_dir, "*.svs"))
print(len(svs_folders))

svs_folder = svs_folders[0]

files = glob(os.path.join(svs_folder, "*.png"))

temp_images = np.asarray([cv2.imread(name) for name in files])

print(temp_images.shape)

norm_model_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/acd_model_weights.pth"

norm_tiles_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/norm_tiles"


normalizer = StainNormalizer(_template_dc_mat=np.array([[ 1.1952132 , -0.5512814 , -0.2789848 ],
       [ 0.35361397,  1.189934  , -0.8477794 ],
       [-0.13746937, -0.10843743,  1.1402813 ]]),_template_w_mat= [np.array(1.5187707, dtype='float32'), np.array(1.888015, dtype='float32'), np.array(1)],
                            model_weights_path = norm_model_path)

#new_images = np.asarray([cv2.imread(name) for name in template_list[:1]])


norm_images =normalizer.transform(temp_images[:5,:,:,[2,1,0]])

with h5py.File(os.path.join(norm_tiles_dir, "normalized_images.h5"), "w") as f:
    f.create_dataset("norm_images", data=norm_images, compression="gzip")

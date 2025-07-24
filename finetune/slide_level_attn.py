import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
from einops import rearrange
import torch.nn.functional as F
import math
from extract_attention import *
import pandas as pd





test_csv_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/train_test_split/test_0.csv"    #train_0.csv
slide_crop_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/ColorJitter_h5_files_test"


#path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-30_gc-5_blr-0.005_wd-0.05_ld-0.95_feat-5-11_save_attention/eval_pretrained_lbd_pat_strat/train_121_1_attention_gradients_last_epoch.pt"
#path =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-10_gc-5_blr-0.005_wd-0.05_ld-0.95_feat-5-11_save_attention_hooks_test_zero_model_grad/eval_pretrained_lbd_pat_strat/test_0_1_attention_gradients_last_epoch.pt"
path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-25_gc-5_blr-0.001_wd-0.005_ld-0.75_feat-5-11_norm_tiles_save_attn/eval_pretrained_lbd_pat_strat/test_0_1_attention_gradients_last_epoch.pt"

data_dict, label,slide_id = load_data_from_csv(test_csv_path, slide_crop_path, 0 )

print(slide_id)

#tile_slide_path = os.path.join(tile_path, slide_id)
coords_np = data_dict['coords'].detach().cpu().numpy()
coords_df = pd.DataFrame(coords_np, columns= ["x","y"])
print(coords_df.head(2))

gradients = torch.load(path)

for k in gradients.keys():
    print(k)

attn_grads=[]
attn = []
attn_product = []

for i in range(10):
    grad = gradients["attn_grad_"+str(i)] 
    #grad = (grad-grad.min())/(grad.max()-grad.min())
    attn_grads.append(torch.from_numpy(grad.reshape(grad.shape[1],16,48)))
    attn_val= gradients["attn_"+str(i)] 
    #attn_val = (attn_val-attn_val.min())/(attn_val.max()-attn_val.min())
    attn.append(torch.from_numpy(attn_val.reshape(attn_val.shape[1],16,48)))
    attn_bmn_grad =  grad*attn_val
    attn_product.append(torch.from_numpy(attn_bmn_grad.reshape(attn_bmn_grad.shape[1],16,48)))
    
    
print(len(attn_grads))
print(attn_grads[0].shape)
    
attn_grads_stacked = torch.stack(attn_grads, axis=0)
attn_stacked = torch.stack(attn, axis=0)
attn_product_stacked =  torch.stack(attn_product, axis=0)
print(attn_grads_stacked.shape)

attn_grads_mean = torch.mean(attn_grads_stacked, axis=3)
attn_mean = torch.mean(attn_stacked, axis=3)
attn_product_mean =  torch.mean(attn_product_stacked, axis=3)
print(attn_grads_mean.shape)

attn_grads_mean = torch.mean(attn_grads_mean, axis=0)
attn_mean = torch.mean(attn_mean, axis=0)
attn_product_mean =  torch.mean(attn_product_mean, axis=0)
print(attn_grads_mean.shape)


df1 = pd.DataFrame(attn_mean.cpu().numpy()[1:,:], columns = ["head"+str(i) for i in range(16)])
df2 = pd.DataFrame(attn_product_mean.cpu().numpy()[1:,:], columns = ["head"+str(i) for i in range(16)])
df3 = pd.DataFrame(attn_grads_mean.cpu().numpy()[1:,:], columns = ["head"+str(i) for i in range(16)])


df1 = pd.concat([coords_df, df1], axis=1, ignore_index=False)
df2 = pd.concat([coords_df, df2], axis=1, ignore_index=False)
df3 = pd.concat([coords_df, df3], axis=1, ignore_index=False)
#print(df.head(2))

df1.to_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-25_gc-5_blr-0.001_wd-0.005_ld-0.75_feat-5-11_norm_tiles_save_attn/"+slide_id+"_attn_mean.csv")
df2.to_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-25_gc-5_blr-0.001_wd-0.005_ld-0.75_feat-5-11_norm_tiles_save_attn/"+slide_id+"_attn_prod.csv")
df3.to_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-25_gc-5_blr-0.001_wd-0.005_ld-0.75_feat-5-11_norm_tiles_save_attn/"+slide_id+"_attn_grad.csv")
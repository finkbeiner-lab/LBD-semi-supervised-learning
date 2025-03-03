import numpy as np
import torch
from sklearn.cluster import KMeans
import os
from glob import glob

def cosine_similarity(tensor1, tensor2):
    return torch.nn.functional.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).item()


a = np.load('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/slide_encoding/PD17_C34-45_Amygdala.svs.npy',allow_pickle=True) 
print(a.item())
print(a.item()["layer_12_embed"].shape)

b = np.load('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/slide_encoding/PD17_Amygdala_C110-115 David Menassa.svs.npy',allow_pickle=True) 
print(b.item()["layer_12_embed"].shape)


tensor1 = a.item()["layer_12_embed"].squeeze(0)
tensor2 = b.item()["layer_12_embed"].squeeze(0)
#print(tensor1.shape)
#tensor1 = torch.randn(768)
#tensor2 = torch.randn(768)
#print(tensor1.shape)

similarity = cosine_similarity(tensor1, tensor2)
print("Cosine Similarity:", similarity)


a = np.load('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/slide_encoding/PD97_C34-45_Amygdala.svs.npy',allow_pickle=True) 
print(a.item()["last_layer_embed"].shape)

b = np.load('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/slide_encoding/PD17_Amygdala_C110-115 David Menassa.svs.npy',allow_pickle=True) 
print(b.item()["last_layer_embed"].shape)


tensor1 = a.item()["last_layer_embed"].squeeze(0)
tensor2 = b.item()["last_layer_embed"].squeeze(0)
#print(tensor1.shape)
#tensor1 = torch.randn(768)
#tensor2 = torch.randn(768)
#print(tensor1.shape)

similarity = cosine_similarity(tensor1, tensor2)
print("Cosine Similarity:", similarity)
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


head_dim = 768 // 16
scaling = head_dim**-0.5
print(scaling)
temperature = 0.1

def get_attention_grads_given_layer(q,k,v, layer):
    q = rearrange(q[layer].squeeze(0), 'b l (h d) -> b h l d', h=16)
    k = rearrange(k[layer].squeeze(0), 'b l (h d) -> b h l d', h=16)
    v = rearrange(v[layer].squeeze(0), 'b l (h d) -> b h l d', h=16)
    q = q*100
    k=k*100
    scores = torch.einsum('b h i d, b h j d -> b h i j', q, k)    #/ math.sqrt(16)
    attn_weights = F.softmax(scores/temperature, dim=-1)
    return scores



def get_attention_weights_given_layer(q,k,v, layer):
    q = rearrange(q[layer].squeeze(0), 'b l (h d) -> b h l d', h=16)
    k = rearrange(k[layer].squeeze(0), 'b l (h d) -> b h l d', h=16)
    v = rearrange(v[layer].squeeze(0), 'b l (h d) -> b h l d', h=16)
    
    q*=scaling
    #q = F.normalize(q, p=2,eps=1e-8, dim=-1)
    #k = F.normalize(k, p=2,eps=1e-8, dim=-1)
    #print("q normalize", q.shape)
    #print("k normalize", k.shape)

    #attn_weights = torch.bmm(q, k.transpose(-2, -1)) 
    #k_transposed = k.transpose(-2, -1)
    #print("k transposed", k_transposed.shape)

    # Compute attention
    scores = torch.einsum('b h i d, b h j d -> b h i j', q, k)    #/ math.sqrt(16)
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = F.dropout(attn_weights, False, training=False)       
    #print(attn_weights.shape)
    #print("----------------------attn_weights---------------------")
    #print(attn_weights[0][0])
    return attn_weights


def plot_heatmap(heatmap, image, save_name):
    grid_size=16
    patch_size =16
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    # 1. Original image with patches
    ax[0].imshow(image)
    ax[0].set_title('Input Patches')
    for i in range(grid_size):
        for j in range(grid_size):
            rect = plt.Rectangle((j*patch_size, i*patch_size), 
                            patch_size, patch_size, 
                            linewidth=1, edgecolor='white', facecolor='none')
            ax[0].add_patch(rect)
    #heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
    for i in range(grid_size):
        for j in range(grid_size):
            rect = plt.Rectangle((j*patch_size, i*patch_size), 
                            patch_size, patch_size, 
                            linewidth=1, edgecolor='white', facecolor='none')
            ax[1].add_patch(rect)
    ax[1].imshow(image)
    ax[1].imshow(heatmap, cmap="jet", alpha=0.3)
    ax[1].set_title('Heatmap')
    plt.savefig(os.path.join(save_path,save_name))
    plt.close()



def grad_rollout(attentions, gradients, discard_ratio):
    result = torch.eye(attentions[0].size(-1))
    #print(result.shape)
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):                
            weights = grad
            attention_heads_fused = (attention*weights).mean(axis=1)
            #print("attention_heads_fused",attention_heads_fused)
            attention_heads_fused[attention_heads_fused < 0] = 0
            #print("attention_heads_fused2",attention_heads_fused)
            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            #print("flat",flat)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            #indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)
    
    #print(result.shape)
    # Look at the total attention between the class token,
    # and the image patches
    mask = result   #[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1))
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    pil_img = Image.fromarray(mask).convert('RGB')
    #pil_img.save("test.png")
    return mask


#path1 = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-30_gc-5_blr-0.005_wd-0.05_ld-0.95_feat-5-11_save_attention/eval_pretrained_lbd_pat_strat/PD171_C110-115_EntCx David Menassa.svs_attention_weights.pt"
#path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-30_gc-5_blr-0.005_wd-0.05_ld-0.95_feat-5-11_save_attention/eval_pretrained_lbd_pat_strat/PD171_C110-115_EntCx David Menassa.svs_attention_weights_qkv.pt"
path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-10_gc-5_blr-0.005_wd-0.05_ld-0.95_feat-5-11_save_attention_hooks_test/eval_pretrained_lbd_pat_strat/test_0_1_attention_gradients_last_epoch.pt"

path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-25_gc-5_blr-0.005_wd-0.005_ld-0.95_feat-5-11_norm_tiles_save_attn/eval_pretrained_lbd_pat_strat/test_0_1_attention_gradients_last_epoch.pt"
path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-25_gc-5_blr-0.001_wd-0.005_ld-0.75_feat-5-11_norm_tiles_save_attn/eval_pretrained_lbd_pat_strat/test_0_1_attention_gradients_last_epoch.pt"
path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-25_gc-5_blr-0.0005_wd-0.0001_ld-0.75_feat-5-11_norm_tiles_save_attn/eval_pretrained_lbd_pat_strat/test_0_1_attention_gradients_last_epoch.pt"


#gradients1 = torch.load(path1)

#print(gradients1["attn_grad_1"])


gradients = torch.load(path)
    
num_heads=16

q_weights = []
k_weights = []
v_weights = []
q_grad_weights = []
k_grad_weights = []
v_grad_weights = []

for i in range(5):
    q_weights.append(torch.from_numpy(gradients["q_"+str(i)].reshape(1,gradients["q_0"].shape[0],16,48)))
    k_weights.append(torch.from_numpy(gradients["k_"+str(i)].reshape(1, gradients["q_0"].shape[0],16,48)))
    v_weights.append(torch.from_numpy(gradients["v_"+str(i)].reshape(1,gradients["q_0"].shape[0],16,48)))
    
    q_grad_weights.append(torch.from_numpy(gradients["q_grad_"+str(i)].reshape(1,gradients["q_0"].shape[0],16,48)))
    k_grad_weights.append(torch.from_numpy(gradients["k_grad_"+str(i)].reshape(1, gradients["q_0"].shape[0],16,48)))
    v_grad_weights.append(torch.from_numpy(gradients["v_grad_"+str(i)].reshape(1,gradients["q_0"].shape[0],16,48)))
    
    
attn_weights = get_attention_weights_given_layer(q_weights,k_weights,v_weights, 0)
attn_grad_weights = get_attention_grads_given_layer(q_grad_weights,k_grad_weights,v_grad_weights, 0)



print("attn_weights", attn_weights.shape)
print("attn_grad_weights",attn_grad_weights.shape)


test_csv_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/train_test_split/test_0.csv"

slide_crop_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/ColorJitter_h5_files_test"
#slide_crop_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/norm_tiles"


save_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-25_gc-5_blr-0.0005_wd-0.0001_ld-0.75_feat-5-11_norm_tiles_save_attn/attention_plots"

tile_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/tiles/output"

data_dict, label,slide_id = load_data_from_csv(test_csv_path, slide_crop_path, 0 )
tile_slide_path = os.path.join(tile_path, slide_id)
coords_np = data_dict['coords'].detach().cpu().numpy()



index_val = 1000

x = str(coords_np[index_val][0]).zfill(5)  # Ensures 5-digit padding
y = str(coords_np[index_val][1]).zfill(5)

img_name = f"{x}x_{y}y"
img_path = os.path.join(tile_slide_path,  img_name+".png")
#save_path = os.path.join(save_attention_path,  "avg_gc_all_lyr_"+img_name+".png")

image = cv2.imread(img_path)  # Read image
print(image.shape)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)  # Convert from BGR to RGB


mask = grad_rollout(attn_weights[index_val], attn_grad_weights[index_val], 0.5)

print(mask.shape)
heatmap_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_CUBIC)

plot_heatmap(heatmap_resized, image, img_name+"1.png")



attn_ = attn_weights[index_val].mean(dim=0).cpu().numpy().astype(np.float32)
print(attn_.shape)
attn_resized = cv2.resize(attn_, (256, 256), interpolation=cv2.INTER_CUBIC)
attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

plot_heatmap(attn_resized, image, img_name+"_attn1.png")

print(attn_grad_weights[index_val])
attn_grad= attn_grad_weights[index_val].mean(dim=0).cpu().numpy().astype(np.float32)

attn_grad_resized = cv2.resize(attn_grad, (256, 256), interpolation=cv2.INTER_CUBIC)
attn_grad_resized = (attn_grad_resized - attn_grad_resized.min()) / (attn_grad_resized.max() - attn_grad_resized.min() + 1e-8)
print(attn_grad_resized)

plot_heatmap(attn_grad_resized, image, img_name+"_attn_grad1.png")

attention_heads_fused = attn_resized*attn_grad_resized

plot_heatmap(attention_heads_fused, image, img_name+"_attn_grad_product1.png")

import torch
import numpy as np
from torch import nn
import sys
import torch
import h5py
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import cv2
from scipy.ndimage import gaussian_filter  # Import the Gaussian filter
import umap
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
#import matplotlib.offsetbox as offsetbox
# For convinience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent))

import time
import torch.utils.tensorboard as tensorboard

from gigapath.classification_head import get_model
from metrics import calculate_metrics_with_task_cfg
from utils import (get_optimizer, get_loss_function, \
                  Monitor_Score, get_records_array,
                  log_writer, adjust_learning_rate)
import math
import torch.nn.functional as F
from einops import rearrange
#import umap.umap_ as umaps
#import umap.plot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics


    
def shuffle_data(images: torch.Tensor, coords: torch.Tensor) -> tuple:
    '''Shuffle the serialized images and coordinates'''
    indices = torch.randperm(len(images))
    images_ = images[indices]
    coords_ = coords[indices]
    return images_, coords_

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
        
def get_images_from_path(img_path: str) -> dict:
    '''Get the images from the path'''
    if '.pt' in img_path:
        images = torch.load(img_path)
        coords = 0
    elif '.h5' in img_path:
        assets, _ = read_assets_from_h5(img_path)
        images = torch.from_numpy(assets['features'])
        coords = torch.from_numpy(assets['coords'])

        # if shuffle the data
        #if self.shuffle_tiles:
        #images, coords = shuffle_data(images, coords)

        if images.size(0) > 1000000:
            images = images[:1000000, :]
        if coords.size(0) > 1000000:
            coords = coords[:1000000, :]
    
    # set the input dict
    data_dict = {'imgs': images,
            'img_lens': images.size(0),
            'pad_mask': 0,
            'coords': coords}
    return data_dict


def select_random_crops(data_dict,n_samples):
    '''Select random crops from the data dictionary'''
    """Select random crops from the data dictionary by randomly sampling indices"""
    # Get total number of crops
    n_crops = data_dict['imgs'].shape[0]
    
    # Randomly sample indices 
    n_samples = min(1000, n_crops) # Cap at 1000 samples
    #indices = torch.randperm(n_crops)[:n_crops- n_samples]
    picked_indices = torch.randperm(n_crops)[:n_samples]
    # Get all indices
    all_indices = torch.arange(n_crops)
    # Get filtered (removed) indices
    filtered_indices = torch.tensor(list(set(all_indices.tolist()) - set(picked_indices.tolist())))

    # Select random crops and their coordinates
    selected_imgs= data_dict['imgs'][filtered_indices]
    selected_coords = data_dict['coords'][filtered_indices]
    
    # Update data dictionary with selected samples
    selected_dict = {
        'imgs': selected_imgs,
        'img_lens': selected_imgs.size(0),
        'pad_mask': 0,
        'coords': selected_coords
    }
    return selected_dict, picked_indices


def load_model(epoch, fold, args):
    model = get_model(**vars(args))
    model = model.to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'fold_' + str(fold), "checkpoint.pt")), strict=True)
    #print(model)
    return model
    


def load_data_from_csv(csv_path, slide_crop_path, idx):
    test_csv = pd.read_csv(csv_path)
    slide_id = test_csv["slide_id"].iloc[idx]
    label =  test_csv["label"].iloc[idx]
    slide_h5_path = os.path.join(slide_crop_path,slide_id.replace(".svs",".h5") )
    data_dict = get_images_from_path(slide_h5_path)
    return data_dict, label,slide_id

def test_evaluate(data_dict, epoch, fold, args):
    model = load_model(epoch, fold, args)
    # set the evaluation records
    #task_setting = args.task_config.get('setting', 'multi_class')
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    with torch.no_grad():
        #for batch_idx, batch in enumerate(loader):
            # load the batch and transform this batch
           # images, img_coords, label = batch['imgs'], batch['coords'], batch['labels']
        images = data_dict['imgs'].to(args.device, non_blocking=True)
        img_coords = data_dict['coords'].to(args.device, non_blocking=True)
        #label = label.to(args.device, non_blocking=True).long()

        with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
            # get the logits
            logits ,dilate_att, q,k,v = model(images, img_coords)
            print(len(q))
            print(q[0].shape)
            print(q[0].squeeze(0).shape)
            
    return dilate_att, q,k,v



def get_attention_weights_given_layer(q,k,v, layer):
    q = rearrange(q[layer].squeeze(0), 'b l (h d) -> b h l d', h=16)
    k = rearrange(k[layer].squeeze(0), 'b l (h d) -> b h l d', h=16)
    v = rearrange(v[layer].squeeze(0), 'b l (h d) -> b h l d', h=16)
    # Compute attention
    scores = torch.einsum('b h i d, b h j d -> b h i j', q, k) / math.sqrt(16)
                
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = F.dropout(attn_weights, False, training=False)       
    print(attn_weights.shape)
    return attn_weights


def rollout(attentions, head_fusion, args):
    # Initialize with identity matrix
    num_patches = attentions[0].shape[-1]
    result = torch.eye(num_patches, dtype = torch.float16).to(args.device)

    for attn in attentions:
        attn = attn.mean(dim=0) if head_fusion == "mean" else attn.max(dim=0)[0]
        attn += torch.eye(attn.size(-1)).to(attn.device) # A + I
        attn /= attn.sum(dim=-1, keepdim=True) # Normalizing A
        result = torch.matmul(result,attn )  # Combine across layers
    return result

"""
def rollout(attentions, discard_ratio, head_fusion, args):
    result = torch.eye(attentions.size(-1)).to(args.device)
    with torch.no_grad():
        if head_fusion == "mean":
            attention_heads_fused = attentions.mean(axis=0)
        elif head_fusion == "max":
            attention_heads_fused = attentions.max(axis=0)[0]
        elif head_fusion == "min":
            attention_heads_fused = attentions.min(axis=0)[0]
        else:
            raise "Attention head fusion type Not supported"

        # Drop the lowest attentions, but
        # don't drop the class token
        flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
        _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
        indices = indices[indices != 0]
        flat[0, indices] = 0

        I = torch.eye(attention_heads_fused.size(-1)).to(args.device)
        a = (attention_heads_fused + 1.0*I)/2
        a = a / a.sum(dim=-1)

        result = torch.matmul(a, result)
    return result
"""




def plot_attention_heatmaps(attention_weights, num_samples=3):
    """
    attention_weights: Tensor of shape [batch_size, num_heads, query_len, key_len]
    num_samples: Number of batch samples to visualize
    """
    # Convert to numpy array if using GPU
    if isinstance(attention_weights, torch.Tensor):
        attn = attention_weights.detach().cpu().numpy()
    else:
        attn = attention_weights

    # Select random samples from batch
    batch_indices = np.random.choice(attn.shape[0], num_samples, replace=False)
    
    for idx in batch_indices:
        fig, axs = plt.subplots(4, 4, figsize=(20, 20))  # For 16 heads (4x4 grid)
        fig.suptitle(f'Attention Heads - Batch Sample {idx}', fontsize=16)
        
        # Plot each head
        for head in range(16):
            row = head // 4
            col = head % 4
            im = axs[row, col].imshow(attn[idx, head], cmap='viridis', vmin=0, vmax=1)
            axs[row, col].set_title(f'Head {head+1}')
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            
        # Add colorbar
        cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig("test.jpg")



def plot_vit_attention_rollout(image_path, attention_weights, save_path):
    image = cv2.imread(image_path)  # Read image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    cropped_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = np.array(cropped_image)  # Convert to NumPy array
    grid_size = 16  # 2
    patch_size = 16
    # Prepare attention matrix
    attn = attention_weights.detach().cpu().numpy()
    heatmap = attn.reshape(grid_size, grid_size)
    # Create visualization
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
    heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
    for i in range(grid_size):
        for j in range(grid_size):
            rect = plt.Rectangle((j*patch_size, i*patch_size), 
                               patch_size, patch_size, 
                               linewidth=1, edgecolor='white', facecolor='none')
            ax[1].add_patch(rect)
    ax[1].imshow(image)
    ax[1].imshow(heatmap_resized, cmap="jet", alpha=0.3)
    #im = ax[1].imshow(heatmap_resized, cmap="jet", alpha=0.6)
    #plt.colorbar(im, ax=ax[1])
    ax[1].set_title('Heatmap')
    plt.savefig(save_path)



def plot_vit_attention(image_path, attention_weights, save_path, all_heads=False):
    """
    ViT-specific attention visualization
    Args:
        image: Input image (224, 224, 3)
        attention_weights: Tensor of shape [batch, heads, num_patches+1, num_patches+1]
        head_idx: Which attention head to visualize
        query_patch: (row, col) of reference patch to highlight
    """
    image = cv2.imread(image_path)  # Read image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    # Center crop to 224x224
    #cropped_image = center_crop(image, 224, 224)
    cropped_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = np.array(cropped_image)  # Convert to NumPy array
    # Calculate patch grid parameters
    patch_size = 16
    grid_size = 16  # 224/16
    num_patches = grid_size ** 2
    
    
    selected_images = []
    
    if all_heads==True:
        for head_idx in range(len(attention_weights)):
            # Prepare attention matrix
            attn = attention_weights[head_idx]
            #.detach().cpu().numpy()
            heatmap = attn.reshape(grid_size, grid_size)
            heatmap =heatmap.astype(np.float32)
            print(heatmap.shape)
            print(heatmap.dtype)
            heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
            selected_images.append(heatmap_resized)
        
        # Create visualization
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.flatten()  # Flatten to iterate easily
        #print(axes)
        
        
        for ax, heatmap_resized in zip(axes, selected_images):
            ax.imshow(image)
            for i in range(grid_size):
                for j in range(grid_size):
                    rect = plt.Rectangle((j*patch_size, i*patch_size), 
                                    patch_size, patch_size, 
                                    linewidth=1, edgecolor='white', facecolor='none')
                    ax.add_patch(rect)
            ax.imshow(heatmap_resized, cmap="jet", alpha=0.3)
        
        # Remove empty subplots if we have less than 50 images
        for ax in axes[len(selected_images):]:
            ax.axis("off")
        
        plt.savefig(save_path)
        
    else:
        #attn = attention_weights.detach().cpu().numpy()
        attn = attention_weights
        heatmap = attn.reshape(grid_size, grid_size)
        heatmap =heatmap.astype(np.float32)
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
        heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        for i in range(grid_size):
            for j in range(grid_size):
                rect = plt.Rectangle((j*patch_size, i*patch_size), 
                                patch_size, patch_size, 
                                linewidth=1, edgecolor='white', facecolor='none')
                ax[1].add_patch(rect)
        ax[1].imshow(image)
        ax[1].imshow(heatmap_resized, cmap="jet", alpha=0.3)
        ax[1].set_title('Heatmap')
        plt.savefig(save_path)
    
def generate_grad_cam(data_dict, epoch,target_class, fold, args):
        # Enable gradient calculation for inputs
    model = load_model(epoch, fold, args)
    model.train()  # Switch to train mode to enable gradients
    
    #for name, param in model.named_parameters():
    #    print(f"{name}: {param.requires_grad}")
        
    #print(model.target_activations.requires_grad)  # Should be True
    # set the evaluation records
    #task_setting = args.task_config.get('setting', 'multi_class')
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    #with torch.no_grad():
    images = data_dict['imgs'].to(args.device, non_blocking=True)
    img_coords = data_dict['coords'].to(args.device, non_blocking=True)
    image = images.clone().requires_grad_(True)
    #coords = img_coords.clone()
    with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
        # get the logits
        logits, _, _, _, _  = model(image, img_coords)
        target = logits[:, target_class]
        model.zero_grad()
        target.backward(retain_graph=True)
        # Get gradients and activations
        # Get gradients and activations
        gradients = model.gradients  # From hook
        activations = model.target_activations.detach()
    
            
            
    
def grad_cam(model, logits,target_class ):
    
    # Get target class (e.g., predicted class)
    target_class = logits.argmax(dim=1).item()
    logit_target = logits[0, target_class]

    # Compute gradients of target logit w.r.t. target_activations
    activations = model.target_activations
    gradients = torch.autograd.grad(
        outputs=logit_target,
        inputs=activations,
        retain_graph=True  # Retain graph for possible reuse
    )[0]  # Shape [N, L, D]
    # Average gradients spatially to get weights
    alpha = gradients.mean(dim=1, keepdim=True)  # [N, 1, D]

    # Weight activations by alpha and sum over features
    heatmap = (alpha * activations).sum(dim=-1)  # [N, L]

    # Post-process heatmap
    heatmap = torch.relu(heatmap)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap

def add_thumbnails_with_labels(ax, umap_embeds, image_paths, labels, label_to_color, image_size=(32, 32), zoom=0.5):
    for i in range(len(image_paths)):
        img = Image.open(image_paths[i])
        img = img.resize(image_size)
        
        # Create annotation box
        imgbox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(img, zoom=zoom), 
            (umap_embeds[i, 0], umap_embeds[i, 1]),
            frameon=False
        )
        ax.add_artist(imgbox)

        # Add label with colored background
        label_text = str(labels[i])
        color = label_to_color[labels[i]]
        ax.text(
            umap_embeds[i, 0], umap_embeds[i, 1] + 0.1,  # Position label slightly above image
            label_text, fontsize=8, weight="bold",
            color="white", ha="center",
            bbox=dict(facecolor=color, edgecolor='black', boxstyle="round,pad=0.3")  # Background color
        )
    
def umap_features(features, target_labels,thumbnail_path_list, args):
    mapper = umap.UMAP(densmap=True,metric="cosine").fit(features)
    fig, ax = plt.subplots(figsize=(16, 12))
    target_label_names = ["DLB" if x==1 else "PDD" for x in target_labels]

    umap.plot.points(mapper, labels=target_labels, color_key_cmap='Paired', ax=ax)
    umap_embeds = mapper.embedding_
    # Define color mapping for labels
    unique_labels = np.unique(target_label_names)
    color_map = plt.get_cmap("Paired", len(unique_labels))  # Generate distinct colors
    label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}
    # Overlay thumbnails on UMAP scatter plot
    add_thumbnails_with_labels(ax, umap_embeds, thumbnail_path_list,target_label_names, label_to_color, image_size=(48, 48), zoom=0.75)
    handles, _ = ax.get_legend_handles_labels()
    label_mapping = {0: "PDD", 1: "DLB"}
    ax.legend(handles, label_mapping.values(), title="LBD type")
    plt.savefig(os.path.join(args.save_dir,"umap_plot.png"), dpi=600, bbox_inches='tight')



# Function to remove a single image without modifying the original dictionary
def remove_single_image(data_dict, index):
    if 0 <= index < len(data_dict['imgs']):
        new_imgs = torch.cat((data_dict['imgs'][:index], data_dict['imgs'][index+1:]))
        new_coords = torch.cat((data_dict['coords'][:index], data_dict['coords'][index+1:]))
        return {'imgs': new_imgs, 'coords': new_coords}  # Return a new dictionary without modifying the original
    else:
        print("Invalid index")
        return data_dict
    

    
    
def get_dilated_attention(model, layer, data_dict, fp16_scaler, args, hook):
    #attn_matrices = []
    #for lyr in range(12):
    print("----attention_map------")
    hook_handle = model.slide_encoder.encoder.layers[layer].self_attn.register_forward_hook(hook)
    with torch.no_grad():
        with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
            _ = model(data_dict['imgs'].to(args.device), data_dict['coords'].to(args.device))  # Run inference
    #return attn_logits
    
def reformat_coords(coord):
    #coord = coord.cpu().numpy()
    x = str(coord[0]).zfill(5)  # Ensures 5-digit padding
    y = str(coord[1]).zfill(5)
    img_name = f"{x}x_{y}y.png"
    return img_name
    coords_list=  [reformat_coords(coord) for coord in list(coords_np)]
    
    
def plot_test_performance_metric(fold,test_csv_path,slide_crop_path,tile_path, len_csv,save_name,args):
    model = load_model( args.epochs, fold, args)
    model.eval()
    fp16_scaler = torch.cuda.amp.GradScaler()
    slide_ids = []
    act_label = []
    pred_prob = []
    pred_labels = []
    for i in range(len_csv):
        data_dict, label,slide_id = load_data_from_csv(test_csv_path, slide_crop_path, i )
        print(label,slide_id)
        tile_slide_path = os.path.join(tile_path, slide_id)
        coords_np = data_dict['coords'].detach().cpu().numpy()
        with torch.no_grad():
            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                logits,h,q,k,v = model(data_dict['imgs'].to(args.device), data_dict['coords'].to(args.device))  # Run inference
                probs = F.softmax(logits, dim=1)
                print(probs)
        baseline_score = probs[0][label].item()
        pred_label = torch.argmax(probs[0])
        print(pred_label)
        pred_labels.append(int(pred_label.cpu().numpy()))
        slide_ids.append(slide_id)
        act_label.append(label)
        pred_prob.append(baseline_score)
    pd.DataFrame({"slide_id":slide_ids,"act_label":act_label,'pred_prob':pred_prob,
                  "pred_label":pred_labels}).to_csv(os.path.join(args.save_dir,save_name))
    
    y_test = [1-v for v in act_label]
    y_prob = [1-v for v in pred_prob]
    fpr1, tpr1, thresholds1 = metrics.roc_curve(y_test,y_prob)
    roc_auc_0 = metrics.roc_auc_score(y_test,y_prob)
    print(roc_auc_0)
    fpr2, tpr2, thresholds2 = metrics.roc_curve(act_label, pred_prob)
    roc_auc_1 = metrics.roc_auc_score(act_label, pred_prob)
    print(roc_auc_1)
    # Print ROC curve
    plt.plot(tpr1,fpr1, label='PDD')
    plt.plot(tpr2,fpr2, label='DLB')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    #plt.savefig(os.path.join(args.save_dir,"roc_curve.png"))
    plt.close()
    cm = confusion_matrix(act_label, pred_labels)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=["PDD","DLB"])
    #disp.plot()
    #cm = plot_confusion_matrix(clf, X , y, cmap=plt.cm.Greens)
    #plt.savefig(os.path.join(args.save_dir,"confusion_matrix1.png"))
    
    
    

def run_ablation_for_each_image(fold,test_csv_path,slide_crop_path, idx,save_ablation_path, args):
    model = load_model( args.epochs, fold, args)
    model.eval()
    fp16_scaler = torch.cuda.amp.GradScaler()
    data_dict, label,slide_id = load_data_from_csv(test_csv_path, slide_crop_path, idx )
    print(label,slide_id)
    tile_slide_path = os.path.join(tile_path, slide_id)
    with torch.no_grad():
        with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
            logits,h,q,k,v = model(data_dict['imgs'].to(device), data_dict['coords'].to(device))  # Run inference
            probs = F.softmax(logits, dim=1)
    #print(probs)
    baseline_score = probs[0][label].item()
    coords_np = data_dict['coords'].detach().cpu().numpy()  
    importance_list = []
    coords_list = []
    pred_label_list = []
    for index in range(len(data_dict['imgs'])):
        data_dict_new = remove_single_image(data_dict, index)
        with torch.no_grad():
            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                logits,_,_,_,_ = model(data_dict_new['imgs'].to(device), data_dict_new['coords'].to(device))  # Run inference
                ablated_probs = F.softmax(logits, dim=1)
        
        pred_label = torch.argmax(ablated_probs)
        drop = baseline_score - ablated_probs[0, label].item()
        importance_list.append(drop)
        coords_list.append(data_dict['coords'][index])
        pred_label_list.append(pred_label.cpu().numpy())

        
    if not os.path.exists(os.path.join(save_ablation_path,slide_id)):
        os.makedirs(os.path.join(save_ablation_path,slide_id))
    
    sorted_indices = np.argsort(importance_list)
    coords_to_use = np.array(coords_list)[sorted_indices]
    coords_x=  [coord[0] for coord in list(coords_np)]
    coords_y=  [coord[1] for coord in list(coords_np)]
    df_save = pd.DataFrame({"idx": list(np.arange(len(coords_list))), "coords":coords_list,"coords_x":coords_x, "coords_y":coords_y, "importance":importance_list, "pred_labels":pred_label_list})
    df_save.to_csv(os.path.join(save_ablation_path,slide_id,slide_id+".csv"))
    df_save = df_save.sort_values(by='importance', ascending=False)
    pos_imp = df_save.head(20)
    df_save["importance"] = -1*df_save["importance"]
    df_save = df_save.sort_values(by='importance', ascending=True)
    neg_imp = df_save.head(20)
    r = get_attention_weights_given_layer(q,k,v, 11)
    
    for i in range(len(neg_imp)):
        img_name = neg_imp.iloc[i]["coords"]
        idx = neg_imp.iloc[i]["idx"]
        source_path = os.path.join(tile_slide_path,img_name)
        destination_path =  os.path.join(save_ablation_path,slide_id,"neg_"+img_name)
        shutil.copy(source_path, destination_path)
        avg_attentions = r[idx].mean(dim=0)
        save_path = os.path.join(save_ablation_path,slide_id,  "neg_mean_layer_11"+"_"+img_name+".png")
        plot_vit_attention(source_path, avg_attentions, save_path)
        
        #Image.save(os.path.join("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/ablationCam_output/", img_name))
    
    for i in range(len(pos_imp)):
        img_name = pos_imp.iloc[i]["coords"]
        idx = pos_imp.iloc[i]["idx"]
        source_path = os.path.join(tile_slide_path,img_name)
        destination_path =  os.path.join(save_ablation_path,slide_id,"pos_"+img_name)
        shutil.copy(source_path, destination_path)
        avg_attentions = r[idx].mean(dim=0)
        save_path = os.path.join(save_ablation_path,slide_id,  "pos_mean_layer_11"+"_"+img_name+".png")
        plot_vit_attention(source_path, avg_attentions, save_path)


def remove_from_particular_data(data_dict, indices_to_remove):
    n_crops = data_dict['imgs'].shape[0]
    all_indices = torch.arange(n_crops)
    # Boolean mask for the cluster to remove
    filtered_indices = torch.tensor([ i for i in range(len(data_dict['coords'])) if i not in indices_to_remove])
    #print("picked_mask", filtered_indices)
    # Get indices NOT in the selected cluster
    #filtered_indices = all_indices[~picked_mask]  
    # Select the filtered data
    selected_imgs = data_dict['imgs'][filtered_indices]
    selected_coords = data_dict['coords'][filtered_indices]
    #print(len(selected_imgs))
    #print(len(selected_coords))
    selected_dict = {
        'imgs': selected_imgs,
        'img_lens': selected_imgs.size(0),
        'pad_mask': 0,
        'coords': selected_coords
    }
    return selected_dict


def ablation_cam_select_random_patches():
    data_dict, label,slide_id = load_data_from_csv(test_csv_path, slide_crop_path, 18 )
    print(label,slide_id)
    tile_slide_path = os.path.join(tile_path, slide_id)
    with torch.no_grad():
        with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
            logits,h,q,k,v = model(data_dict['imgs'].to(device), data_dict['coords'].to(device))  # Run inference
            probs = F.softmax(logits, dim=1)
    baseline_score = probs[0][label].item()
    coords_np = data_dict['coords'].detach().cpu().numpy()

    def select_random_patches(tile_coordinates, num_patches, patch_size, tile_size,min_wsi_width, max_wsi_width, min_wsi_height, max_wsi_height):
        #Select random patches where all tiles are consecutive in a grid.
        selected_patches = []
        
        for _ in range(num_patches):
            #while True:
                # Choose a random starting point that allows a full patch to fit
                
            random_coord = random.choice(tile_coordinates)
            x_start = int(random_coord[0])
            y_start = int(random_coord[1])
            
            patch_tiles = [[x_start + i * tile_size, y_start + j * tile_size]
                        for j in range(patch_size) for i in range(patch_size)]
            
            if all(tile in tile_coordinates for tile in patch_tiles):
                selected_patches.append(patch_tiles)
                    #for tile in patch_tiles:
                    #    tile_coordinates.remove(tile)
            #break  # Successfully found a valid patch
        return selected_patches


    min_wsi_width, max_wsi_width = min([x[0] for x in coords_np]), max([x[0] for x in coords_np])
    min_wsi_height, max_wsi_height = min([x[1] for x in coords_np]), max([x[1] for x in coords_np])
    
    num_patches = 20  # Number of random patches to select
    selected_patches = select_random_patches(coords_np, num_patches, 4, 512,min_wsi_width, max_wsi_width, min_wsi_height, max_wsi_height)
    
    print(data_dict['imgs'])
    print(data_dict['imgs'][0])
    
    print(len(selected_patches))
    
    df1= pd.DataFrame()
    
    for i, patch in enumerate(selected_patches):
        print(len(patch))
        print(patch[0])
        filtered_dict= {'imgs':[],'coords':[]}
        for img, coord in zip(data_dict["imgs"], data_dict['coords']):
            y = coord.cpu().numpy() 
            if not any(np.array_equal(coord, p) for p in patch):    
            #if all(coord not in patch):
                filtered_dict["imgs"].append(img)
                filtered_dict["coords"].append(coord)
            
        filtered_dict["imgs"] = torch.tensor(np.stack(filtered_dict["imgs"]))
        filtered_dict["coords"] = torch.tensor(np.stack(filtered_dict["coords"]))
       #new_data_dict = remove_from_particular_data(data_dict.copy(), patch)
        print(len(filtered_dict['imgs']))
        #print(filtered_dict['imgs'])
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                logits,h,q,k,v = model(filtered_dict['imgs'].to(device), filtered_dict['coords'].to(device))  # Run inference
                ablated_probs = F.softmax(logits, dim=1)
                print(ablated_probs)
        drop = baseline_score - ablated_probs[0, label].item()
        print(drop)
        coords_x = [ x[0] for x in patch]
        coords_y = [ x[1] for x in patch]
        df = pd.DataFrame({"coords_x":coords_x,"coords_y":coords_y})
        df["cluster"] = i
        df["imp_score"] = drop
        if len(df1)==0:
            df1 = df
        else:
            df1 = pd.concat([df1, df], axis=0)
    df1.to_csv(os.path.join(save_ablation_path,slide_id+"_ablation_cam.csv"))



def select_random_patches(tile_coordinates, num_patches, patch_size, tile_size,min_wsi_width, max_wsi_width, min_wsi_height, max_wsi_height):
    #Select random patches where all tiles are consecutive in a grid.
    selected_patches = []
    
    for _ in range(num_patches):
        #while True:
            # Choose a random starting point that allows a full patch to fit
            
        #random_coord = random.choice(tile_coordinates)
        random_coord = tile_coordinates[0]
        x_start = int(random_coord[0])
        y_start = int(random_coord[1])
        
        patch_tiles = [[x_start + i * tile_size, y_start + j * tile_size]
                    for j in range(patch_size) for i in range(patch_size)]
        
        if all(tile in tile_coordinates for tile in patch_tiles):
            selected_patches.append(patch_tiles)
                #for tile in patch_tiles:
                #    tile_coordinates.remove(tile)
        #break  # Successfully found a valid patch
    return selected_patches

def get_tiles_in_region(tile_coords, region):
    #Get tiles that are contained within a given region.
    #Parameters:
    #    tile_coords (list of tuples): List of (x, y) coordinates of tiles.
    #    region (tuple): (x_start, y_start, x_end, y_end) defining the region.
    #Returns:
    #    list of tuples: Tiles that fall within the given region.
    x_start, y_start, x_end, y_end = region
    # Filter tiles that are inside the region
    contained_tiles = [
        [x, y] for x, y in tile_coords if x_start <= x < x_end and y_start <= y < y_end
    ]
    return contained_tiles


def get_regions(coords_np, region_size):
    min_x, min_y = min([x[0] for x in coords_np]), min([x[1] for x in coords_np])
    max_x, max_y = max([x[0] for x in coords_np]), max([x[1] for x in coords_np])
    print( min_x, min_y, max_x, max_y)
    #region_size = 2048*4*4*2
    # Compute region grid start and end
    x_start_grid = min_x//region_size
    y_start_grid = min_y//region_size 
    x_end_grid = max_x//region_size 
    y_end_grid = max_y//region_size 
    print(x_start_grid,y_start_grid,x_end_grid,  y_end_grid)
    regions = []
    for region_x in range(x_start_grid, x_end_grid + 1):
        for region_y in range(y_start_grid, y_end_grid + 1):
            x_start = region_x * region_size
            y_start = region_y * region_size
            x_end = x_start + region_size
            y_end = y_start + region_size
            regions.append((x_start, y_start, x_end, y_end))
    print(len(regions))
    return regions




def get_ablation_on_regions(regions, data_dict, region_size, flag="sequential"):
    selected_patches = []
    if flag=="sequential":
        regions  = get_regions(coords_np, region_size)
        
        for region in regions:
            contained_tiles = get_tiles_in_region(coords_np, region)
            selected_patches.append(contained_tiles)
            print(len(contained_tiles))
        #print(len(selected_patches))
    else:
        num_patches = 20  # Number of random patches to select
        selected_patches = select_random_patches(coords_np, num_patches, 32, 512,min_x, max_x, min_y, max_y)
        
    df1= pd.DataFrame()
    for i, patch in enumerate(selected_patches):
        selected_patches_set = {tuple(p) for p in patch}
        #print(len(selected_patches_set))
        filtered_dict = {
        "imgs": [img for img, coord in zip(data_dict["imgs"], data_dict["coords"]) 
                if tuple(coord.cpu().numpy()) not in selected_patches_set],
        "coords": [coord for coord in data_dict["coords"] 
                if tuple(coord.cpu().numpy()) not in selected_patches_set]
        }
        imgs_f = np.stack(filtered_dict["imgs"])  # Shape: (100, 512)
        coords_f = np.stack(filtered_dict["coords"]) 
        # Convert to PyTorch tensor
        img_ft = torch.tensor(imgs_f, dtype=torch.float16)
        coords_f_ft = torch.tensor(coords_f)

        with torch.no_grad():
            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                logits,h,q,k,v = model(img_ft.to(device),coords_f_ft.to(device))  # Run inference
                ablated_probs = F.softmax(logits, dim=1)
                print(ablated_probs)
        #drop = baseline_score - ablated_probs[0, label].item()
        drop = max(0, baseline_score - ablated_probs[0, label].item())

        print(drop)
        coords_x = [ x[0] for x in patch]
        coords_y = [ x[1] for x in patch]
        df = pd.DataFrame({"coords_x":coords_x,"coords_y":coords_y})
        df["cluster"] = i
        df["imp_score"] = drop
        if len(df1)==0:
            df1 = df
        else:
            df1 = pd.concat([df1, df], axis=0)
        
    df1.to_csv(os.path.join(save_ablation_path,slide_id+"_"+str(region_size)+"_region_ablation_cam.csv"))
    
 
 
 
def get_hooks_from_trained_models():
    data_dict, label, slide_id = load_data_from_csv(test_csv_path, slide_crop_path, 0)
    label = torch.tensor([label], dtype=torch.int64).to(device)
    print(slide_id)
    attention_gradients = {}

    def save_attention_gradients(name):
        def hook(module, grad_input, grad_output):
            attention_gradients[name] = grad_output[0].detach().cpu().numpy()  # Store gradients
        return hook
    
    def register_attention_hooks(model):
        for i, layer in enumerate(model.slide_encoder.encoder.layers[:10]):  # Track first 10 layers
            #layer.self_attn.q_proj.register_full_backward_hook(save_attention_gradients(f"attn_grad_{i}"))
            #layer.self_attn.q_proj.register_forward_hook(save_attention_gradients(f"attn_q_{i}"))
            #layer.self_attn.q_proj.register_full_backward_hook(save_attention_gradients(f"attn_grad_q_{i}"))
            #layer.self_attn.k_proj.register_forward_hook(save_attention_gradients(f"attn_k_{i}"))
            #layer.self_attn.k_proj.register_full_backward_hook(save_attention_gradients(f"attn_grad_k_{i}"))
            #layer.self_attn.v_proj.register_forward_hook(save_attention_gradients(f"attn_v_{i}"))
            #layer.self_attn.v_proj.register_full_backward_hook(save_attention_gradients(f"attn_grad_v_{i}"))
            layer.self_attn.register_forward_hook(save_attention_gradients(f"attn_{i}"))
            layer.self_attn.register_full_backward_hook(save_attention_gradients(f"attn_grad_{i}"))
    loss_fn = get_loss_function(args.task_config)
    coords_np = data_dict['coords'].detach().cpu().numpy()
    
    num_indices = 5
    
    random_inds = random.sample(range(0, len(coords_np)), num_indices)
    
    discard_pct = 80
    
    tile_slide_path = os.path.join(tile_path, slide_id)
    
    model = load_model( args.epochs, fold, args)
    for name, param in model.slide_encoder.named_parameters():
        print(name, param.requires_grad)
    model.eval()
    model.zero_grad()
    
    images = data_dict['imgs'].to(args.device, non_blocking=True)
    img_coords = data_dict['coords'].to(args.device, non_blocking=True)
    label = torch.tensor([label]).to(args.device, non_blocking=True).long()
    
    with torch.enable_grad():  # Allow gradients during validation
        with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
            # get the logits
            logits,h,q,k,v = model(images, img_coords,  args.register_hook)
            # get the loss
            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                label = label.squeeze(-1).float()
            else:
                label = label.squeeze(-1).long()
        
            if torch.isnan(logits).any():
                print("NaNs in logits!")
            
            loss = loss_fn(logits[0], label)
            fp16_scaler.scale(loss).backward()
            
            torch.save(attention_gradients, os.path.join(args.save_dir, slide_id+"_attention_weights.pt"))
            #torch.save(attention_gradients, os.path.join(args.save_dir, slide_id+"_attention_weights_qkv.pt"))
 
    # load gradients
    path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-30_gc-5_blr-0.005_wd-0.05_ld-0.95_feat-5-11_save_attention/eval_pretrained_lbd_pat_strat/test_0_1_attention_gradients_last_epoch.pt"
    
    gradients = torch.load(path)
    
    for k in gradients.keys():
        print(gradients[k])
 
 
    
    

# Usage Example
# Assuming:
# - image: Your 224x224 input image (numpy array)
# - attn_weights: Your attention tensor [batch, heads, 197, 197] (196 patches + class token)

# Visualize global attention for head 3


# Visualize attention from center patch (row=7, col=7)
#plot_vit_attention(image, attn_weights, head_idx=3, query_patch=(7,7))

if __name__ == "__main__":
    # Initialize model and CAM generator
    #model = YourGigapathModel()  # Your actual model implementation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    #checkpoint_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-30_blr-0.002_wd-0.05_ld-0.95_feat-11/eval_pretrained_lbd/fold_0/checkpoint.pt"
    checkpoint_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-30_blr-0.002_wd-0.05_ld-0.95_feat-5-11/eval_pretrained_lbd_pat_strat/fold_0/checkpoint.pt"
    test_csv_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/train_test_split/lbd_pat_strat/test_0.csv"
    #test_csv_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/train_test_split/lbd/test_new.csv"
    slide_crop_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/ColorJitter_h5_files_test"
    #ColorJitter_h5_files_test"
    #slide_crop_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/h5_files"
    tile_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/tiles/output"
    save_ablation_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/ablationCam_output"
    save_attention_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/attention_results/colorjitter_data_dr_1_seg_1024"
    
    #clustered_dab_csv_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/antibody_pdd_dlb_clustering/slide_cluster_dab_tiles.csv"
    #clustered_dab_csv_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/antibody_pdd_dlb_clustering/slide_cluster_pdd_dlb_tiles.csv"

    print(device)
    #slide_id = test_csv["slide_id"].iloc[0]
    
    model = load_model_from_checkpoint(device, checkpoint_path)
        
    print(model)
        
    model.eval()
        
    fp16_scaler = torch.cuda.amp.GradScaler()
    
    data_dict, label,slide_id = load_data_from_csv(test_csv_path, slide_crop_path, 0 )
        
    tile_slide_path = os.path.join(tile_path, slide_id)
    
    
        
      

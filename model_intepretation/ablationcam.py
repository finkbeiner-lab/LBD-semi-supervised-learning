import torch
import numpy as np
from torch import nn
import sys
sys.path.append('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath')
from gigapath.classification_head import get_model
import matplotlib.pyplot as plt
from finetune.params import get_finetune_params
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

def remove_from_particular_cluster(data_dict, cluster_num):
    n_crops = data_dict['imgs'].shape[0]
    all_indices = torch.arange(n_crops)
    # Boolean mask for the cluster to remove
    filtered_indices = torch.tensor([ i for i in range(len(data_dict['cluster'])) if data_dict['cluster'][i] != cluster_num])
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
    



def load_model_from_checkpoint(device, checkpoint_path):
    args = get_finetune_params()
    model = get_model(**vars(args))
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint)
    print(model.slide_encoder.encoder.layers[-1].self_attn)
    print(model.slide_encoder.encoder.layers[-1].final_layer_norm)
    return model


def load_data_from_csv(csv_path, slide_crop_path, idx):
    test_csv = pd.read_csv(csv_path)
    slide_id = test_csv["slide_id"].iloc[idx]
    label =  test_csv["label"].iloc[idx]
    slide_h5_path = os.path.join(slide_crop_path,slide_id.replace(".svs",".h5") )
    data_dict = get_images_from_path(slide_h5_path)
    return data_dict, label,slide_id

def hook(module, input, output):
    print(f"Input shape: {[i.shape for i in input]}")
    #print(f"Output shape: {output.shape}")
    print(len(output))
    print(output[0].shape)
    print(len(output[1]))
    #print(output)
    global attn_logits
    attn_logits = output[0]  # Storing output
    
def hook_slide(module, input, output):
    print(f"Input shape: {[i.shape for i in input]}")
    #print(f"Output shape: {output.shape}")
    print(len(output))
    print(output[0].shape)
    #print(len(output[1]))
    #print(output)
    global slide_features
    slide_features = output[0]  # Storing output


def plot_images(image_paths, save_dir, fig_name, num_images=20, rows=4, cols=5):
    """
    Plots 'num_images' images from 'image_paths' in a grid of 'rows' x 'cols'
    """
    selected_images = image_paths  # Select first 'num_images'
    #selected_images = random.sample(image_paths, num_images)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 9))
    axes = axes.flatten()  # Flatten to iterate easily

    for ax, img_path in zip(axes, selected_images):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")  # Hide axis
    
    # Remove empty subplots if we have less than 50 images
    for ax in axes[len(selected_images):]:
        ax.axis("off")
    
    #plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(save_dir,fig_name))


def center_crop(img, target_height, target_width):
    """
    Center crops an image using OpenCV.

    Args:
        img (numpy.ndarray): Input image as a NumPy array (H, W, C).
        target_height (int): Desired crop height.
        target_width (int): Desired crop width.

    Returns:
        numpy.ndarray: Cropped image.
    """
    h, w, _ = img.shape  # Get original dimensions
    # Compute center
    center_y, center_x = h // 2, w // 2
    # Compute cropping box
    crop_x1 = max(center_x - target_width // 2, 0)
    crop_x2 = min(center_x + target_width // 2, w)
    crop_y1 = max(center_y - target_height // 2, 0)
    crop_y2 = min(center_y + target_height // 2, h)
    # Perform cropping
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    return cropped_img


def visualize_attention(image_path, attn_weights, save_path, alpha=0.5, cmap="jet"):
    """
    Visualizes attention weights over an image.

    Args:
        image_path (str): Path to the image file.
        attn_weights (numpy array): Attention map (H', W').
        alpha (float): Transparency level for blending.
        cmap (str): Colormap for heatmap.
    """
    # Load and convert image to numpy with transformation
    #transform = transforms.Compose([
    #    transforms.CenterCrop(224),  # Crops the image to 224x224
    #])
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = np.array(image)  # Convert to NumPy array
    print("Image shape:", image.shape)

    # Normalize attention weights to range [0, 1]
    attn_weights = attn_weights - attn_weights.min()
    attn_weights = attn_weights / attn_weights.max()

    # Resize attention map to match image size (224x224 after crop)
    attn_resized = Image.fromarray(np.uint8(attn_weights * 255))  # Convert to uint8 and scale to [0, 255]
    attn_resized = attn_resized.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
    attn_resized = np.array(attn_resized) / 255.0  # Normalize back to [0, 1]
    print("Resized attention map shape:", attn_resized.shape)
    
    attn_resized = gaussian_filter(attn_resized, sigma=1)
    print("After Gaussian filter shape:", attn_resized.shape)

    # Plot the original image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")

    # Overlay attention map on top of the image
    plt.imshow(attn_resized, cmap=cmap, alpha=alpha)

    # Save the figure before showing it
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # Optionally show the plot (commented out)
    # plt.show()
    plt.close()

    
    
def display_attention_only(image_path, attention_map, save_path,alpha=0.5, cmap="jet"):
    image = cv2.imread(image_path)  # Read image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    # Center crop to 224x224
    #cropped_image = center_crop(image, 224, 224)
    cropped_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    height, width = 24, 32  # Adjust based on model's spatial dimensions
    attention_2d = attention_map.reshape(height, width)
    # Normalize
    attention_normalized = (attention_2d - attention_2d.min()) / (attention_2d.max() - attention_2d.min())
    attention_normalized = np.uint8(attention_normalized * 255)
    ## Upsample to image size (e.g., 224x224)
    upsampled_attention = cv2.resize(attention_normalized, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6)) 
    axes[0].imshow(cropped_image)
    axes[0].axis("off")
    axes[0].set_title("Original Image")
    # Attention Overlay
    axes[1].imshow(cropped_image)
    axes[1].imshow(upsampled_attention, cmap=cmap, alpha=alpha)
    axes[1].axis("off")
    axes[1].set_title("Attention Overlay")
    #plt.axis("off")
    #plt.imshow(image)
    #plt.imshow(upsampled_attention,  cmap=cmap, alpha=alpha)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
def compute_rollout(attn_matrices, head_fusion="mean", discard_ratio=0.9):
    # Initialize with identity matrix
    #print(attn_matrices[0].size())
    #print(attn_matrices[0].shape())
    print(len(attn_matrices[0]))
    #rollout = torch.eye(len(attn_matrices)).to(attn_matrices[0].device)
    rollout =torch.full(len(attn_matrices[0]), 1).to(attn_matrices[0].device)
    
    print(rollout.shape)

    for attn in reversed(attn_matrices):  # Process from last layer to first
        # Add residual connection (identity matrix)
        attn = attn + torch.full(len(attn_matrices[0]), 1).to(attn_matrices[0].device)
        attn = attn / attn.sum(dim=-1, keepdim=True)  # Renormalize
        
        # Recursively multiply with previous rollout
        rollout = torch.matmul(attn, rollout)

    print(rollout.shape)
    # Aggregate over the class token (if present) or take mean
    if discard_ratio < 1.0:
        # Discard low-attention regions (optional)
        flat_rollout = rollout.view(-1)
        indices = flat_rollout.argsort(descending=True)
        retain_len = int(flat_rollout.size(0) * (1 - discard_ratio))
        mask = torch.zeros_like(flat_rollout)
        mask[indices[:retain_len]] = 1
        rollout = rollout * mask.view(rollout.shape)
    

    return rollout.mean(dim=0)  # Average over batch (B=1)


def compute_1d_rollout(attn_maps, discard_ratio=0.2):
    # Initialize with uniform weights
    rollout = torch.ones_like(attn_maps[0])
    
    for attn in reversed(attn_maps):
        # Rescale attention to [0,1] and apply multiplicative fusion
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        rollout *= attn  # Element-wise multiplication
    print(rollout.shape)
    # Optional: Discard low-activation regions
    if discard_ratio > 0:
        flat = rollout.flatten()
        flat= flat.float()
        print(flat)
        threshold = torch.quantile(flat, discard_ratio)
        rollout[rollout < threshold] = 0
    
    return rollout




def find_rolling_attention(data_dict,fp16_scaler, model):
    coords_np = data_dict['coords'].detach().cpu().numpy()
    for i in range(20):
        test_coords = coords_np[i]
        x = str(test_coords[0]).zfill(5)  # Ensures 5-digit padding
        y = str(test_coords[1]).zfill(5)
        img_name = f"{x}x_{y}y"
        attn_matrices = []
        for lyr in range(5):
            print("----attention_map------")
            hook_handle = model.slide_encoder.encoder.layers[lyr].self_attn.register_forward_hook(hook)
            with torch.no_grad():
                with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                    _ = model(data_dict['imgs'].to(device), data_dict['coords'].to(device))  # Run inference
            
            print("layer ", lyr)
            #print(attn_logits)
            
            attn_probs = torch.exp(attn_logits)  
            attn_probs /= attn_probs.sum(dim=-2, keepdim=True)  # Normalize
            attn_matrices.append(attn_logits[0][i])
            hook_handle.remove()
        #attn_img = attn_logits[0][0]
        #print(attn_img)
        
        #attn_matrices2.append(attn_logits[0][1])
            # Remove hook after checking
        #display_attention_only(os.path.join(tile_slide_path, img_name+".png"), attn_img, os.path.join(save_attention_path, img_name+"_"+str(lyr)+".png"),alpha=0.3)
        #cnt = cnt+1
        #print(len(attn_matrices))
        rollout_attention =  compute_1d_rollout(attn_matrices, 0.2)
        display_attention_only(os.path.join(tile_slide_path, img_name+".png"), rollout_attention.cpu().numpy(), os.path.join(save_attention_path, img_name+"_"+"rollout"+".png"),alpha=0.2)
    

# Example usage
if __name__ == "__main__":
    # Initialize model and CAM generator
    #model = YourGigapathModel()  # Your actual model implementation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #checkpoint_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-30_blr-0.002_wd-0.05_ld-0.95_feat-11/eval_pretrained_lbd/fold_0/checkpoint.pt"
    checkpoint_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-30_blr-0.002_wd-0.05_ld-0.95_feat-5-11/eval_pretrained_lbd_pat_strat/fold_0/checkpoint.pt"
    test_csv_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/train_test_split/lbd_pat_strat/test_0.csv"
    #test_csv_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/train_test_split/lbd/test_new.csv"
    slide_crop_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/DAB_ColorJitter_h5_files"
    #ColorJitter_h5_files_test"
    #slide_crop_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/h5_files"
    tile_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/tiles/output"
    save_ablation_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/ablationCam_output"
    save_attention_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/attention_results/DAB_colorjitter_data"
    
    #clustered_dab_csv_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/antibody_pdd_dlb_clustering/slide_cluster_dab_tiles.csv"
    #clustered_dab_csv_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/antibody_pdd_dlb_clustering/slide_cluster_pdd_dlb_tiles.csv"

    print(device)
    #slide_id = test_csv["slide_id"].iloc[0]
    
    model = load_model_from_checkpoint(device, checkpoint_path)
        
    print(model)
        
    model.eval()
        
    fp16_scaler = torch.cuda.amp.GradScaler()
    
    """
    hook_handle = model.slide_encoder.encoder.layer_norm.register_forward_hook(hook_slide)
    
    data_dict, label,slide_id = load_data_from_csv(test_csv_path, slide_crop_path, 0)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
            _ = model(data_dict['imgs'].to(device), data_dict['coords'].to(device))  # Run inference

    print(slide_features.shape)
    
    """
    for idx in range(25):
        data_dict, label,slide_id = load_data_from_csv(test_csv_path, slide_crop_path, idx)
        
        tile_slide_path = os.path.join(tile_path, slide_id)
        
        print(slide_id, label)
        with torch.no_grad():
            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                logits = model(data_dict['imgs'].to(device), data_dict['coords'].to(device))
                probs = F.softmax(logits, dim=1)
                
        baseline_score = probs[0, label].item()
        print(probs)
    #print("atten_weights:", atten_weights)
    #print(attn_weights)
    #print(baseline_score)
    
    """
    importance_list = []
    coords_list = []
    pred_label_list = []
    for i in range(50):
        selected_dict, picked_indices = select_random_crops(data_dict, 20)
        with torch.no_grad():
            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                logits = model(selected_dict['imgs'].to(device), selected_dict['coords'].to(device))
                ablated_probs = F.softmax(logits, dim=1)
        
        pred_label = torch.argmax(ablated_probs)
        drop = baseline_score - ablated_probs[0, label].item()
        #print(drop)
        #if pred_label==0:
        importance_list.append(drop)
        coords_list.append(data_dict['coords'][picked_indices])
        pred_label_list.append(pred_label.cpu().numpy())
        
    #print(pred_label_list)
    #print(importance_list)#
    #print(coords_list)
    
    sorted_indices = np.argsort(importance_list)
    
    
    coords_to_use = np.array(coords_list)[sorted_indices]
    
    
    for i in range(len(coords_to_use)):
        c = coords_to_use[i]
        paths_to_plot =[]
        for tile_x, tile_y in c:
                # Proper zero-padding
            x = str(tile_x).zfill(5)  # Ensures 5-digit padding
            y = str(tile_y).zfill(5)

            img_name = f"{x}x_{y}y.png"
            paths_to_plot.append(os.path.join(tile_slide_path, img_name))
        print(importance_list[sorted_indices[i]])
        plot_images(paths_to_plot, save_ablation_path, "ablation_cam_"+str(i))
        #break
  
    
    
    
    
    
    #test_coords = coords_np[0]
    #x = str(test_coords[0]).zfill(5)  # Ensures 5-digit padding
    #y = str(test_coords[1]).zfill(5)
    #img_name = f"{x}x_{y}y"
    #attn_matrices1 = []
    #attn_matrices2 = []
    
    
    
    """
   
    #slide_features =  model.slide_encoder(data_dict['imgs'].to(device), data_dict['coords'].to(device))
    
    #print(slide_features)
    
    
    """ Get attention
    for ind in [18, 0, 3,4,20,21,18]:
        data_dict, label,slide_id = load_data_from_csv(test_csv_path, slide_crop_path, ind)
        #print(data_dict.keys(),data_dict)
        tile_slide_path = os.path.join(tile_path, slide_id)
        coords_np = data_dict['coords'].detach().cpu().numpy()
        for i in range(20):
            test_coords = coords_np[i]
            x = str(test_coords[0]).zfill(5)  # Ensures 5-digit padding
            y = str(test_coords[1]).zfill(5)
            img_name = f"{x}x_{y}y"
            attn_matrices = []
            for lyr in range(12):
                print("----attention_map------")
                hook_handle = model.slide_encoder.encoder.layers[lyr].self_attn.register_forward_hook(hook)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                        _ = model(data_dict['imgs'].to(device), data_dict['coords'].to(device))  # Run inference
                
                print("layer ", lyr)
                #print(attn_logits)
                
                attn_probs = torch.exp(attn_logits)  
                attn_probs /= attn_probs.sum(dim=-2, keepdim=True)  # Normalize
                attn_matrices.append(attn_logits[0][i])
                hook_handle.remove()
        #attn_img = attn_logits[0][0]
            #print(attn_img)
            
            #attn_matrices2.append(attn_logits[0][1])
            # Remove hook after checking
            #display_attention_only(os.path.join(tile_slide_path, img_name+".png"), attn_img, os.path.join(save_attention_path, img_name+"_"+str(lyr)+".png"),alpha=0.3)
            #cnt = cnt+1
            #print(len(attn_matrices))
            rollout_attention =  compute_1d_rollout(attn_matrices, discard_ratio=0.2)
            display_attention_only(os.path.join(tile_slide_path,  img_name+".png"), rollout_attention.cpu().numpy(), os.path.join(save_attention_path, slide_id + "_" +img_name+"_"+"rollout"+".png"),alpha=0.4)
    """
    
    """
    rollout_attention1 =  compute_1d_rollout(attn_matrices1, discard_ratio=-2)
    rollout_attention2 =  compute_1d_rollout(attn_matrices2, discard_ratio=-2)
    print(rollout_attention1)
    print(rollout_attention1.shape)
    display_attention_only(os.path.join(tile_slide_path, img_name+".png"), rollout_attention1.cpu().numpy(), os.path.join(save_attention_path, img_name+"_"+"rollout"+".png"),alpha=0.2)
    test_coords = coords_np[1]
    x = str(test_coords[0]).zfill(5)  # Ensures 5-digit padding
    y = str(test_coords[1]).zfill(5)
    img_name = f"{x}x_{y}y"
    display_attention_only(os.path.join(tile_slide_path, img_name+".png"), rollout_attention2.cpu().numpy(), os.path.join(save_attention_path, img_name+"_"+"rollout"+".png"),alpha=0.2)
   
    file_names = []
    attention_weights = []
    
    cnt = 0
    
    for i, tiles in enumerate(data_dict['coords'].detach().cpu().numpy()):
            # Proper zero-padding
        x = str(tiles[0]).zfill(5)  # Ensures 5-digit padding
        y = str(tiles[1]).zfill(5)
        #print(tiles)
        #print(attn_probs[0][i])
        img_name = f"{x}x_{y}y.png"
        #file_names.append(img_name)
        attn_img = attn_probs[0][i].cpu().numpy()
        #print(img_name)
        print(attn_probs[0][i])
        #visualize_attention(os.path.join(tile_slide_path, img_name), attn_img, os.path.join(save_attention_path, img_name), alpha=0.3, cmap="jet")
        #display_attention_only(attn_img, os.path.join(save_attention_path, "att_test.png"))
        display_attention_only(os.path.join(tile_slide_path, img_name), attn_img, os.path.join(save_attention_path, img_name),alpha=0.2)
        cnt = cnt+1
        if cnt>20:
            break
        
        #break
    """
    #dataset_df = pd.DataFrame({"img_name":file_names, "att_weight":attention_weights})
    
    """
        
    #print(dataset_df.head(2))
    
    #dataset_df.to_csv(os.path.join(save_attention_path,"test.csv"))
    
    
    
    #attn_map = get_attention_map(model, data_dict, 6, 0)
    
    #print("----attention_map------")
    #print(attn_map)
    

    #wsi_cam = WSIAblationCAM(model, target_layer='aggregate.attention')
    
    # Generate CAM
    #heatmap = wsi_cam.generate_cam('path/to/slide.h5', output_shape=(2000, 2000))
    
    # Visualize
    #plt.figure(figsize=(20, 20))
    #plt.imshow(heatmap, cmap='jet', alpha=0.5)
    #plt.colorbar()
    #plt.show()
    
    data_list = []
    
    for ind in range(22):
        
        data_dict, label,slide_id = load_data_from_csv(test_csv_path, slide_crop_path, ind)
        
        print(data_dict["img_lens"])
        
        print(slide_id)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                logits = model(data_dict['imgs'].to(device), data_dict['coords'].to(device))
                probs = F.softmax(logits, dim=1)
                    
        baseline_score = probs[0, label].item()
        
        print(baseline_score)
        
        
        df =  pd.read_csv(clustered_dab_csv_path)
        df["filename_2"] = df["filename"].apply(lambda l: l.replace("b'","").replace("'","").split("/")[-2])
        df["img_name"] = df["filename"].apply(lambda l: l.replace("b'","").replace("'","").split("/")[-1])
        df=df[df["filename_2"]==slide_id]
        print(len(df))
        
        img_names = []
        clusters = []
        for i, tiles in enumerate(data_dict['coords'].detach().cpu().numpy()):
                # Proper zero-padding
            x = str(tiles[0]).zfill(5)  # Ensures 5-digit padding
            y = str(tiles[1]).zfill(5)
            #print(tiles)
            #print(attn_probs[0][i])
            img_name = f"{x}x_{y}y.png"
            img_names.append(img_name)
            clstr = df[df["img_name"]==img_name]['cluster'].values[0]
            clusters.append(clstr)
            
        clusters = [int(x) for x in clusters]
        data_dict["img_names"] = img_names
        data_dict["cluster"] = clusters
        #print(clusters)
        print("clusters",len(clusters))
        #print("inside function")
        for c in range(6):
            selected_dict = remove_from_particular_cluster(data_dict, c)
            print(selected_dict["img_lens"])
            with torch.no_grad():
                with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                    logits = model(selected_dict['imgs'].to(device), selected_dict['coords'].to(device))
                    probs = F.softmax(logits, dim=1)
            new_score = probs[0, label].item()
            print("Cluster",c, new_score, new_score-baseline_score )
            
            data = {'slide': slide_id, 'label': label, 'baseline_score':baseline_score, 'cluster':c,"cluster_importance_score":new_score,"score_diff":new_score-baseline_score }
            data_list.append(data)
    
    final_df = pd.DataFrame(data_list)
    final_df.to_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-30_blr-0.002_wd-0.05_ld-0.95_feat-11/clusterwise_importance.csv")
    
    #print(df)
    #img_names2 = list(df[df["cluster"]==1]["img_name"].values)
    #k = [ l for l in img_names2 if l in img_names]
    #print("match", k)
    """
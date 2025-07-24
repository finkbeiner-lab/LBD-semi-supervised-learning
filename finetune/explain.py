import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
#from training import train, evaluate
from training_attn_hooks import train
from params import get_finetune_params
from task_configs.utils import load_task_config
from utils import seed_torch, get_exp_code, get_splits, get_loader, save_obj, get_loss_function
from datasets.slide_datatset import SlideDataset
import torch.nn.functional as F
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import entropy
import random
from extract_attention import *
from attention_graph_util import *
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_loss_function



def model_training(args):
    dataset = pd.read_csv(args.dataset_csv) # read the dataset csv file
    # use the slide dataset
    DatasetClass = SlideDataset
    # set up the results dictionary
    results = {}
    # start cross validation
    for fold in range(args.folds):
        # set up the fold directory
        save_dir = os.path.join(args.save_dir, f'fold_{fold}')
        os.makedirs(save_dir, exist_ok=True)
        # get the splits
        train_splits, val_splits, test_splits = get_splits(dataset, fold=fold, **vars(args))
        # instantiate the dataset
        train_data, val_data, test_data = DatasetClass(dataset, args.root_path, train_splits, args.task_config, split_key=args.split_key) \
                                        , DatasetClass(dataset, args.root_path, val_splits, args.task_config, split_key=args.split_key) if len(val_splits) > 0 else None \
                                        , DatasetClass(dataset, args.root_path, test_splits, args.task_config, split_key=args.split_key) if len(test_splits) > 0 else None
        args.n_classes = train_data.n_classes # get the number of classes
        # get the dataloader
        train_loader, val_loader, test_loader = get_loader(train_data, val_data, test_data, **vars(args))
        # start training
        val_records, test_records = train((train_loader, val_loader, test_loader), fold, args)
        # update the results
    
        records = {'val': val_records, 'test': test_records}
        for record_ in records:
            for key in records[record_]:
                if 'prob' in key or 'label' in key:
                    continue
                key_ = record_ + '_' + key
                if key_ not in results:
                    results[key_] = []
                results[key_].append(records[record_][key])
    # save the results into a csv file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.save_dir, 'summary.csv'), index=False)
    # print the results, mean and std
    for key in results_df.columns:
        print('{}: {:.4f} +- {:.4f}'.format(key, np.mean(results_df[key]), np.std(results_df[key])))
    print('Results saved in: {}'.format(os.path.join(args.save_dir, 'summary.csv')))
    print('Done!')
    
    #return attention_gradients



def get_raw_attention(discard_pct,random_inds, layers, tile_slide_path,q,k,v,args ):
    patch_size = 16
    grid_size = 16 
    num_patches = grid_size ** 2
    for index_val in random_inds:
        x = str(coords_np[index_val][0]).zfill(5)  # Ensures 5-digit padding
        y = str(coords_np[index_val][1]).zfill(5)
        img_name = f"{x}x_{y}y"
        img_path = os.path.join(tile_slide_path,  img_name+".png")
        image = cv2.imread(img_path)  # Read image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        cropped_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = np.array(cropped_image)  # Convert to NumPy array
        
        for layer in layers:
            att1 = get_attention_weights_given_layer(q,k,v, layer)
            att_lyrs_np = att1[index_val+1].detach().cpu().numpy()
            attention_weights = att_lyrs_np
            print(attention_weights.shape)
            for head_idx in range(len(attention_weights)):
                # Prepare attention matrix
                attn = attention_weights[head_idx]
                attn = (attn - attn.min()) / (attn.max() - attn.min())
                threshold = np.percentile(attn, discard_pct)
                attn1 = np.where(attn > threshold, attn, 0)
                heatmap =attn1.astype(np.float32)
                heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
                os.makedirs(os.path.join(args.save_dir,"raw_attention", slide_id, img_name, "lyr"+str(layer)), exist_ok=True)
                save_path = os.path.join(args.save_dir,"raw_attention", slide_id, img_name,  "lyr"+str(layer), img_name+"_"+str(head_idx)+".png")
                fig, ax = plt.subplots(1, 2, figsize=(20, 8))
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
                ax[1].imshow(heatmap_resized, cmap="jet", alpha=0.3)
                ax[1].set_title('Heatmap')
                plt.savefig(save_path)
                plt.close()
                
                
def get_raw_attention_head_fusion(discard_pct,random_inds, layers, tile_slide_path,q,k,v, agg, args ):
    patch_size = 16
    grid_size = 16 
    num_patches = grid_size ** 2
    for index_val in random_inds:
        x = str(coords_np[index_val][0]).zfill(5)  # Ensures 5-digit padding
        y = str(coords_np[index_val][1]).zfill(5)
        img_name = f"{x}x_{y}y"
        img_path = os.path.join(tile_slide_path,  img_name+".png")
        image = cv2.imread(img_path)  # Read image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        cropped_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = np.array(cropped_image)  # Convert to NumPy array
        
        for layer in layers:
            att1 = get_attention_weights_given_layer(q,k,v, layer)
            att_lyrs_np = att1[index_val+1].detach().cpu().numpy()
            if agg=="MEAN":
                attention_weights = att_lyrs_np.mean(axis=0)
            if agg=="MIN":
                attention_weights = att_lyrs_np.min(axis=0)
            if agg=="MAX":
                attention_weights = att_lyrs_np.max(axis=0)

            attn = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
            threshold = np.percentile(attn, discard_pct)
            attn1 = np.where(attn > threshold, attn, 0)
            heatmap =attn1.astype(np.float32)
            heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
            os.makedirs(os.path.join(args.save_dir,"raw_attention", slide_id, img_name, "lyr"+str(layer)), exist_ok=True)
            save_path = os.path.join(args.save_dir,"raw_attention", slide_id, img_name,  "lyr"+str(layer), img_name+"_"+agg+".png")
            fig, ax = plt.subplots(1, 2, figsize=(20, 8))
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
            ax[1].imshow(heatmap_resized, cmap="jet", alpha=0.3)
            #ax[1].set_title('Heatmap')
            plt.savefig(save_path)
            plt.close()
           


def get_rollout_attention(discard_pct,random_inds, num_layers, tile_slide_path, start_layer,q,k,v, args ):
    patch_size = 16
    grid_size = 16 
    num_patches = grid_size ** 2
    for index_val in random_inds:
        x = str(coords_np[index_val][0]).zfill(5)  # Ensures 5-digit padding
        y = str(coords_np[index_val][1]).zfill(5)
        img_name = f"{x}x_{y}y"
        img_path = os.path.join(tile_slide_path,  img_name+".png")
        image = cv2.imread(img_path)  # Read image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        cropped_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = np.array(cropped_image)  # Convert to NumPy array
        all_layer_attentions = []
        for i in range(num_layers):
            att1 = get_attention_weights_given_layer(q,k,v, i)
            att_lyrs_np = att1[index_val+1].detach().cpu().numpy()
            avg_heads = att_lyrs_np.sum(axis=1) / att_lyrs_np.shape[1]
            all_layer_attentions.append(avg_heads)
        
        num_tokens = all_layer_attentions[0].shape[1]
        eye = np.eye(num_tokens)
        all_layer_attentions = [all_layer_attentions[i] + eye for i in range(len(all_layer_attentions))]
        matrices_aug = [ all_layer_attentions[i] / all_layer_attentions[i].sum(axis=-1, keepdims=True) for i in range(len(all_layer_attentions))]
        joint_attention = matrices_aug[start_layer]
        print(joint_attention.shape)
        
        for i in range(start_layer + 1, len(matrices_aug)):
            joint_attention = matrices_aug[i] @ joint_attention   
            
        #print(joint_attention)
        
        #rollout_explanation = joint_attention[0, 1:].reshape((3, 5))
        #print(rollout_explanation.shape)
        threshold = np.percentile(joint_attention, discard_pct)
        attn1 = np.where(joint_attention > threshold, joint_attention, 0)
        heatmap =attn1.astype(np.float32)
        heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
        os.makedirs(os.path.join(args.save_dir,"rollout_attention", slide_id, img_name), exist_ok=True)
        save_path = os.path.join(args.save_dir,"rollout_attention", slide_id, img_name, img_name+".png")
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
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
        ax[1].imshow(heatmap_resized, cmap="jet", alpha=0.3)
        ax[1].set_title('Heatmap')
        plt.savefig(save_path)
        plt.close()



def get_attention_flow(q,k,v, args):
    att_layers =[]
    for i in range(12):
        att1 = get_attention_weights_given_layer(q,k,v, i)
        att_lyrs_np = att1[random_inds[0]+1].detach().cpu().numpy()
        att_layers.append(att_lyrs_np)
        
    attn_maps = np.stack(att_layers) 
    
    attn_maps = attn_maps.mean(axis=1)
    
    joint_attention = compute_joint_attention(attn_maps, add_residual=False)
    
    print(joint_attention.shape)  # (n_layers, seq_len, seq_len)
    
    input_tokens = [str(i) for i in range(attn_maps.shape[1])]
    
    print(input_tokens)
    
    adj_matrix, labels_to_index = get_adjmat(joint_attention, input_tokens)

    print(adj_matrix.shape)
    # Draw attention flow graph
    G = draw_attention_graph(adj_matrix, labels_to_index, attn_maps.shape[0], attn_maps.shape[1])
    
    print(G)
    
    input_nodes = [str(i) + "_" + input_tokens[i] for i in range(len(input_tokens))]
    
    print(input_nodes)
    
    flow_values = compute_flows(G, labels_to_index, input_nodes, attn_maps.shape[1])

    print(flow_values.shape)
    
    print(joint_attention[0])
    print(flow_values[0])
    
    flow_att_mat = convert_adjmat_tomats(flow_values, n_layers=attn_maps.shape[0], l=attn_maps.shape[-1])
    
    print(flow_att_mat[7])
    
    os.makedirs(os.path.join(args.save_dir,"attention_flow", slide_id, "1"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,"joint_attention", slide_id, "1"), exist_ok=True)
    grid_size=16
    patch_size = 16
    index_val = random_inds[0]
    x = str(coords_np[index_val][0]).zfill(5)  # Ensures 5-digit padding
    y = str(coords_np[index_val][1]).zfill(5)
    img_name = f"{x}x_{y}y"
    img_path = os.path.join(tile_slide_path,  img_name+".png")
    image = cv2.imread(img_path)  # Read image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    cropped_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = np.array(cropped_image)  # Convert to NumPy array
    
    for lyr in range(flow_att_mat.shape[0]):
        save_path = os.path.join(args.save_dir,"joint_attention", slide_id, "1", "0"+"_"+str(lyr)+".png")
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        heatmap_resized = cv2.resize(joint_attention[lyr], (224, 224), interpolation=cv2.INTER_CUBIC)
        for i in range(grid_size):
            for j in range(grid_size):
                rect = plt.Rectangle((j*patch_size, i*patch_size), 
                                patch_size, patch_size, 
                                linewidth=1, edgecolor='white', facecolor='none')
                ax.add_patch(rect)
        #ax[0].imshow(image)
        ax.imshow(image)
        ax.imshow(heatmap_resized, cmap="jet", alpha=0.3)
        ax.set_title('Heatmap')
        plt.savefig(save_path)
        plt.close()
    
    
    for lyr in range(flow_att_mat.shape[0]):
        save_path = os.path.join(args.save_dir,"attention_flow", slide_id, "1", "0"+"_"+str(lyr)+".png")
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        heatmap_resized = cv2.resize(flow_att_mat[lyr], (224, 224), interpolation=cv2.INTER_CUBIC)
        for i in range(grid_size):
            for j in range(grid_size):
                rect = plt.Rectangle((j*patch_size, i*patch_size), 
                                patch_size, patch_size, 
                                linewidth=1, edgecolor='white', facecolor='none')
                ax.add_patch(rect)
        ax.imshow(image)
        ax.imshow(heatmap_resized, cmap="jet", alpha=0.3)
        ax.set_title('Heatmap')
        plt.savefig(save_path)
        plt.close()




if __name__ == '__main__':
    args = get_finetune_params()
    print(args)

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(device)
    
    args.device = device

    # set the random seed
    seed_torch(device, args.seed)

    # load the task configuration
    print('Loading task configuration from: {}'.format(args.task_cfg_path))
    args.task_config = load_task_config(args.task_cfg_path)
    print(args.task_config)
    args.task = args.task_config.get('name', 'task')
    
    # set the experiment save directory
    args.save_dir = os.path.join(args.save_dir, args.task, args.exp_name)
    args.model_code, args.task_code, args.exp_code = get_exp_code(args) # get the experiment code
    args.save_dir = os.path.join(args.save_dir, args.exp_code)
    os.makedirs(args.save_dir, exist_ok=True)
    print('Experiment code: {}'.format(args.exp_code))
    print('Setting save directory: {}'.format(args.save_dir))

    # set the learning rate
    eff_batch_size = args.batch_size * args.gc
    if args.lr is None or args.lr < 0:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.gc)
    print("effective batch size: %d" % eff_batch_size)

    # set the split key
    if args.pat_strat:
        args.split_key = 'pat_id'
    else:
        args.split_key = 'slide_id'
        
   
    #model_training(args)
    
    #saved_gradients = torch.load(os.path.join(args.save_dir, "attention_gradients_last_epoch.pt"))
    
    #print(saved_gradients.keys())
    
    #grad_layer_0 = saved_gradients["attn_grad_0"]
    #print(grad_layer_0.shape)  # Check dimensions
    
    #print(len(attention_gradients))
    
    #print(attention_gradients[0].shape)
    
    fp16_scaler = torch.cuda.amp.GradScaler()
    fold = 0
    save_dir = os.path.join(args.save_dir, f'fold_{fold}')
    args.n_classes = 2
    
    test_csv_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/train_test_split/lbd_pat_strat/test_0.csv"
    #test_csv_path ="/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/train_test_split/normalized_test.csv"

    tile_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/tiles/output"
    #slide_crop_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/ColorJitter_h5_files_test"
    #slide_crop_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/norm_tiles"
    slide_crop_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/dinov2_g14_imperial_colorJitter_h5files"

    test_csv  =  pd.read_csv(test_csv_path).iloc[22]

    #plot_test_performance_metric(fold,test_csv_path,slide_crop_path,tile_path, 19,"norm_pred.csv",args)
    
    data_dict, label, slide_id = load_data_from_csv(test_csv_path, slide_crop_path, 22) #18
    
    
    brain_regions_dict ={'EntCx':0,'Amygdala':1,'Striatum':2, 'Hippo':3}
    antibody_dict ={'C110-115':0,'C34-45':1}
    brainbank_dict ={'Imperial':0,'Oxford':1}
    
    
    
    antibody = antibody_dict[test_csv["anitibody"]]
    brainbank = brainbank_dict[test_csv["Brain_bank"]]
    brainReg = brain_regions_dict[test_csv["brain_region"]]
    
    
    attrs = torch.tensor([brainReg, antibody, brainbank ])
    print(attrs)
    
    label = torch.tensor([label], dtype=torch.int64).to(device)
    
    model = load_model( args.epochs, fold, args)
    
    model.eval()
    
    images = data_dict['imgs'].to(args.device, non_blocking=True)
    img_coords = data_dict['coords'].to(args.device, non_blocking=True)
    label = torch.tensor([label]).to(args.device, non_blocking=True).long()
    attrs = attrs.to(args.device, non_blocking=True)
    
    with torch.no_grad():  # Allow gradients during validation
        with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
            # get the logits
            logits,h,q,k,v = model(images, img_coords, attrs, args.register_hook)
    
    
    
    
    
    """
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
            #layer.self_attn.q_proj.register_full_backward_hook(save_attention_gradients(f"attn_q_{i}"))
            #layer.self_attn.q_proj.register_forward_hook(save_attention_gradients(f"attn_grad_q_{i}"))
            #layer.self_attn.k_proj.register_full_backward_hook(save_attention_gradients(f"attn_k_{i}"))
            #layer.self_attn.k_proj.register_forward_hook(save_attention_gradients(f"attn_grad_k_{i}"))
            #layer.self_attn.v_proj.register_full_backward_hook(save_attention_gradients(f"attn_v_{i}"))
            #layer.self_attn.v_proj.register_forward_hook(save_attention_gradients(f"attn_grad_v_{i}"))
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
       
    #register_attention_hooks(model)   
    
    #optimizer = get_optimizer(args, model)
    
    #optimizer.zero_grad()     # clear old gradients
    
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
            
            #category_mask = torch.zeros(logits.size()).to(args.device, non_blocking=True).long()   
            #print(q[0].shape)
            #logits[:, 0] = 1 
            
            #print("logits", logits)
            
            #print("category_mask",category_mask)
            
            #print("logits*category_mask",logits*category_mask)
            
            #loss = (output*category_mask).sum()
            # Debug gradients
            if torch.isnan(logits).any():
                print("NaNs in logits!")
            
            loss = loss_fn(logits[0], label)
            fp16_scaler.scale(loss).backward()
            
            #torch.save(attention_gradients, os.path.join(args.save_dir, slide_id+"_attention_weights.pt"))
            #torch.save(attention_gradients, os.path.join(args.save_dir, slide_id+"_attention_weights_qkv.pt"))
     
    print(attention_gradients)
    
    
    #model.to(device)
    model.eval()
    
    print(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
            logits,h,q,k,v = model(data_dict['imgs'].to(device), data_dict['coords'].to(device),  register_hook=True) # Run inference
            probs = F.softmax(logits, dim=1) 
            #loss = loss_fn(logits, label)
    """
    #
    
    #get_rollout_attention(discard_pct,random_inds, 10, tile_slide_path, 0,q,k,v, args )



    # Call the function to visualize flow
    #visualize_flow(flow_values, labels_to_index)
    
    
    
    
    
    
    
    
    
    
    #get_raw_attention(discard_pct,random_inds, [0,4,8], tile_slide_path,q,k,v,args )
    #get_raw_attention_head_fusion(discard_pct,random_inds, [0,4,8], tile_slide_path,q,k,v, "MEAN", args )
    #get_raw_attention_head_fusion(discard_pct,random_inds,  [0,4,8], tile_slide_path,q,k,v, "MAX", args )
    #get_raw_attention_head_fusion(discard_pct,random_inds,  [0,4,8], tile_slide_path,q,k,v, "MIN", args )
    
    # Unfreeze all parameters (or unfreeze only the attention layers)
    #for param in model.parameters():
    #    param.requires_grad = True  # Critical for hooks to work
    
    #for name, param in model.slide_encoder.named_parameters():
    #    print(name, param.requires_grad)
    
    
    #for param in model.parameters():
    #    param.requires_grad = True
    
    #model.train()
    
    #for name, param in model.slide_encoder.named_parameters():
    #    print(name, param.requires_grad)
    """
    model.train()
    loss_fn = get_loss_function(args.task_config)
    
    attention_gradients=[]
    #with torch.with():
    with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
        logits,h,q,k,v = model(data_dict['imgs'].to(device).requires_grad_(True), data_dict['coords'].to(device),  register_hook=True) # Run inference
        probs = F.softmax(logits, dim=1) 
        loss = loss_fn(logits, label)
        
    loss.backward()
    
    
    for name, param in model.named_parameters():
        if 'attn' in name:  # Assuming 'attn' is part of the layer names for attention weights
            attention_gradients.append(param.grad.cpu().numpy())  # Store gradients as numpy arrays
      
    print(len(attention_gradients))
    print(attention_gradients[0].shape)      
 

    
    att_layers =[]
    for i in range(12):
        att1 = get_attention_weights_given_layer(q,k,v, i)
        att_lyrs_np = att1.detach().cpu().numpy()
        att_layers.append(att_lyrs_np)
        
    attn_maps = np.stack(att_layers) 
    
    #attn_maps = attn_maps.mean(axis=1)
    
    print(attn_maps.shape)
    
    start_layer = 5
    b, h, s, _ = attn_maps[0].shape
    num_blocks = len(attns)
    states = np.mean(attns[-1], axis=1)[:, 0, :].reshape((b, 1, s))
    for i in range(start_layer, num_blocks - 1)[::-1]:
        attn = np.mean(attns[i], 1)
        states_ = states
        states = states @ attn
        states += states_
    
    total_gradients = np.zeros((b, h, s, s))
    
    for alpha in np.linspace(0, 1, steps):
        # forward propagation
        data_scaled = data * alpha
        _, gradients, _ = predict_fn(data_scaled, label=label)
    """
    
    
    
    
    
    
            
    """
    b = data_dict['imgs'].shape[0]
    
    print("b", b)

    kwargs = {"alpha": 1}
    #if index == None:
    index = np.argmax(logits.cpu().data.numpy(), axis=-1)
    
    print("index", index)

    one_hot = np.zeros((b, logits.size()[-1]), dtype=np.float32)
    
    one_hot[np.arange(b), index] = 1
    
    print(one_hot.shape)

        
    one_hot_vector = one_hot
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits)
    
    
    model.zero_grad() 
    
    one_hot.backward(retain_graph=True)
    
    print(model)
    print(model.slide_encoder.encoder.layers[-1].self_attn.get_attention_map())
    
    b, h, s, _ =  model.slide_encoder.encoder.layers[11].self_attn.get_attention_map().shape
    """
    
    
    #one_hot = np.zeros((b, logits.size()[-1]), dtype=np.float32)
    #print(one_hot.shape)
    
    
    
   
    
    #get_raw_attention(discard_pct,random_inds, [0,4,11], tile_slide_path, args )
    #get_rollout_attention(70,random_inds, 7, tile_slide_path, 0, args )

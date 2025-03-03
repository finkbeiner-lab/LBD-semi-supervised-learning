import umap
import torch
import numpy as np
from torch import nn
import sys
sys.path.append('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath')
from gigapath.classification_head import get_model
import matplotlib.pyplot as plt
from finetune.params import get_finetune_params
from ablationcam import load_model_from_checkpoint


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-30_blr-0.002_wd-0.05_ld-0.95_feat-5-11/DAB_colorJitter/eval_pretrained_lbd_pat_strat/fold_0/checkpoint.pt"
model = load_model_from_checkpoint(device, checkpoint_path)

model.eval()
        
fp16_scaler = torch.cuda.amp.GradScaler()

hook_handle = model.slide_encoder.encoder.layers[lyr].self_attn.register_forward_hook(hook)



data = np.random.rand(100)

# Apply UMAP
reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2)
embedding = reducer.fit_transform(data)

print(embedding)
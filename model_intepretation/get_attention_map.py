import torch
import matplotlib.pyplot as plt
import numpy as np

# Function to extract attention map
def get_attention_map(model, image_tensor, layer_idx=0, head_idx=0):
    """
    Extracts attention maps from a specified layer and head.

    Args:
        model (LongNetViT): The LongNetViT model.
        image_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W).
        layer_idx (int): Transformer layer index to extract attention from.
        head_idx (int): Attention head index (0 to num_heads - 1).

    Returns:
        attn_map (numpy array): Extracted attention map.
    """

    # Forward pass to get intermediate activations
    with torch.no_grad():
        outputs = model.forward_features(image_tensor)  # Extract features

    # Extract attention weights from the desired layer
    attn_weights = model.LongNetEncoder.layer[layer_idx].self_attn.attn_weights  # Shape: [batch, heads, seq_len, seq_len]
    
    # Convert to numpy
    attn_map = attn_weights[0, head_idx].cpu().numpy()  # Selecting batch=0, specific head

    return attn_map

# Visualize the attention map
def plot_attention(attn_map, title="Attention Map"):
    plt.figure(figsize=(8, 6))
    plt.imshow(attn_map, cmap="viridis", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Query Position")
    plt.ylabel("Key Position")
    plt.show()




# Example usage
if __name__ == "__main__":
    # Initialize model and CAM generator
    #model = YourGigapathModel()  # Your actual model implementation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = get_finetune_params()
    print(args)
    model = get_model(**vars(args))
    model = model.to(device)
    #print(model)
    
    checkpoint = torch.load("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/finetune/lbd/run_epoch-25_blr-0.002_wd-0.05_ld-0.95_feat-11/eval_pretrained_lbd/fold_0/checkpoint.pt")
    print(checkpoint.keys())
    model.load_state_dict(checkpoint)
    # Example Usage
    # Assuming `model` is a trained LongNetViT and `image_tensor` is a preprocessed input image
    layer_idx = 6  # Choose a deeper layer for global features
    head_idx = 2   # Choose an attention head
    attn_map = get_attention_map(model, image_tensor, layer_idx, head_idx)
    plot_attention(attn_map, title=f"Attention Map (Layer {layer_idx}, Head {head_idx})")

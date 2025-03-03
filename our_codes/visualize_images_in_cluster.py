import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from glob import glob
import os 
from PIL import Image
import random
import pandas as pd

def plot_images(image_paths, cluster_num, save_dir, fig_name, num_images=15, rows=3, cols=5):
    """
    Plots 'num_images' images from 'image_paths' in a grid of 'rows' x 'cols'
    """
    selected_images = image_paths[:num_images]  # Select first 'num_images'
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

"""
def plot_cluster_images(image_paths, cluster_num, fig_name, num_samples=5):
    #cluster_images = [img for img, lbl in zip(image_paths, labels) if lbl == cluster_num][:num_samples]
    cluster_images = image_paths[:num_samples]
    fig, axes = plt.subplots(1, len(cluster_images), figsize=(12, 4))
    for ax, img_path in zip(axes, cluster_images):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")
    #plt.show()
    plt.savefig(os.path.join("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/Semi-supervised-learning/clustering_figures",fig_name))
"""
# View images from cluster 0
"""
#output_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/tile_clustering/tile_clusters.csv"
#output_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/tile_clustering/tile_clusters_pca.csv"
#output_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/tile_clustering/tile_clusters_pca_all_features.csv"
output_path ="/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/antibody_tile_clustering/tile_clusters_pca_all_features.csv"

#k_=10
#k_=4 
k_=5
tile_cluster = pd.read_csv(output_path)
for k in range(k_):
    tmp = tile_cluster[tile_cluster["cluster"]==k]
    image_paths =tmp['filename'].values
    random.shuffle(image_paths)
    print(len(image_paths))
    plot_images(image_paths,k, "cluster_pca_all_features_"+str(k)+".png")
"""


if __name__ == '__main__':       
    #output_path ="/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/antibody_pdd_dlb_clustering/slide_cluster_pdd_dlb_tiles.csv"
    output_path ="/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/antibody_pdd_dlb_clustering/slide_cluster_dab_tiles.csv"

    tile_cluster = pd.read_csv(output_path)
    save_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/antibody_pdd_dlb_clustering"
    k_ = 6
    for k in range(k_):
        tmp = tile_cluster[tile_cluster["cluster"]==k]
        image_paths =tmp['filename'].values
        image_paths= [x.replace("b'", "").replace("'","") for x in image_paths]
        random.shuffle(image_paths)
        print(len(image_paths))
        plot_images(image_paths,k, save_dir, "cluster_dab_features_"+str(k)+".png")
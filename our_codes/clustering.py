import numpy as np
import torch
from sklearn.cluster import KMeans
import os
from glob import glob
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import h5py


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


def find_optimal_clusters(X):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 10)
    # Fit K-means for different values of k
    for k in K:
        kmeanModel = KMeans(n_clusters=k, init = "k-means++", random_state=42).fit(X)
        # Calculate distortion as the average squared distance from points to their cluster centers
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / X.shape[0])
        # Inertia is calculated directly by KMeans
        inertias.append(kmeanModel.inertia_)
        # Store the mappings for easy access
        mapping1[k] = distortions[-1]
        mapping2[k] = inertias[-1]
        print("Distortion values:")
    for key, val in mapping1.items():
        print(f'{key} : {val}')
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.grid()
    plt.savefig("elbow_curve.png")

 
def apply_kmeans_clustering(X,npy_paths, k):
    clusters = KMeans(n_clusters=k, init = "k-means++", random_state=42).fit_predict(X)
    #print([x.split("/")[-1] for x in npy_paths])
    print(clusters)
    df = pd.DataFrame({"filename":[x for x in npy_paths],"cluster":clusters})
    return df
    
 
 
if __name__ == '__main__':       

    #clustering with tiles
    #npy_paths = glob(os.path.join('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/slide_encoding',"*.npy"))
    """ Syn1 data clustering
    PATH = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/slide_encoding/Syn1_data'
    npy_paths = glob(os.path.join(PATH,"*.npy"))


    features = []
    for pth in npy_paths:
        a = np.load(pth,allow_pickle=True) 
        features.append(a.item()["layer_12_embed"].squeeze(0))
        
    features_np = np.array(features)
    #find_optimal_clusters(features_np)
    df = apply_kmeans_clustering(features, npy_paths, 2)

    #df["antibody"] = df["filename"].apply(lambda l : "C34-45" if l.find("C34-45")!=-1 else "C110-115")
    #df["brain_region"] = df["filename"].apply(lambda l : "Amygdala" if l.find("Amygdala")!=-1 else "EntCx")
    #print(df.groupby(["antibody","cluster"])["filename"].count())
    #print(df.groupby(["brain_region","cluster"])["filename"].count())
    df.to_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/slide_cluster_syn1.csv")

    #kmeans = KMeans(n_clusters=3, random_state=15)
    #clusters = kmeans.fit_predict(features)
    #print([x.split("/")[-1] for x in npy_paths])
    #print(clusters)
    #df = pd.DataFrame({"filename":[x.split("/")[-1] for x in npy_paths],"cluster":clusters})
    #print(df)
    """
    
    """ Run PDD/DLB custering
    
    dir_path =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/h5_files"
    
    h5_paths = glob(os.path.join(dir_path,"*.h5"))
    
    print(len(h5_paths))

    features_all = []
    crop_names_all = []
    #len(h5_paths)
    for i in range(len(h5_paths)):
        assets, attrs = read_assets_from_h5(h5_paths[i])
        crp_names =  assets['crop_names']
        features = assets['features']
        
        features_all.append(features)
        crop_names_all.append(crp_names)
        
    print(len(features_all))
    
    features_np = np.concatenate(features_all)
    crop_names_np = np.concatenate(crop_names_all)
    #find_optimal_clusters(features_np)
    df = apply_kmeans_clustering(features_np,crop_names_np, 6)
    df.to_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/antibody_pdd_dlb_clustering/slide_cluster_pdd_dlb_tiles.csv")
    """
    feature_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/aggregate_embeddings/test_data/DLB-007-2015-Amygdala-C34-45.svs/features.npy"
    cropnames_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/aggregate_embeddings/test_data/DLB-007-2015-Amygdala-C34-45.svs/crop_names.npy"

    features_np = np.load(feature_path,allow_pickle=True) 
    cropnames_np = np.load(cropnames_path,allow_pickle=True) 
        
    print(features_np.shape)
    print(cropnames_np.shape)
    
    #find_optimal_clusters(features_np)
    df = apply_kmeans_clustering(features_np,cropnames_np, 6)
    df.to_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/antibody_pdd_dlb_clustering/slide_cluster_dab_tiles.csv")

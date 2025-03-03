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

def find_optimal_clusters(X, nmax):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, nmax)
    # Fit K-means for different values of k
    for k in K:
        kmeanModel = KMeans(n_clusters=k, init = "k-means++",max_iter=500).fit(X) #random old = 42,  random_state=24
        # Calculate distortion as the average squared distance from points to their cluster centers
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / X.shape[0])
        # Inertia is calculated directly by KMeans
        inertias.append(kmeanModel.inertia_)
        # Store the mappings for easy access
        mapping1[k] = distortions[-1]
        mapping2[k] = inertias[-1]
        #print("Distortion values:")
        
    for key, val in mapping1.items():
        print(f'{key} : {val}')
        
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.grid()
    plt.savefig("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/antibody_tile_clustering/elbow_curve_tiles_pca.png")
    
    
def apply_kmeans_clustering(X,npy_paths, k):
    clusters = KMeans(n_clusters=k, init = "k-means++", random_state=42).fit_predict(X)
    #print([x.split("/")[-1] for x in npy_paths])
    print(clusters)
    #df = pd.DataFrame({"filename":[x.split("/")[-1] for x in npy_paths],"cluster":clusters})
    df = pd.DataFrame({"filename":npy_paths,"cluster":clusters})
    return df
    
#pca_features_all = np.load('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/syn1_all_features.npy', allow_pickle=True) 
#image_paths_all = np.load('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/syn1_all_crop_paths.npy',  allow_pickle=True) 

#pca_features = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/syn1_pca_features.npy' 
#image_paths_all = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/syn1_image_paths.npy'

#pca_features = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/syn1_pca_features_all.npy' 
#image_paths_all = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/syn1_pca_crop_paths_all.npy'

pca_features ='/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/'+'antibody_data_pca_features.npy' 
image_paths_all = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/'+'antibody_data_image_paths.npy'


npy_paths = glob(os.path.join(pca_features,"*.npy"))

pca_features_all = np.load(pca_features,allow_pickle=True) 
image_paths_all = np.load(image_paths_all,allow_pickle=True) 
print(len(pca_features_all))
print(len(image_paths_all))
print(pca_features_all.shape)
print(image_paths_all)
find_optimal_clusters(pca_features_all, 10)
df = apply_kmeans_clustering(pca_features_all, image_paths_all, 4)

#print(df)

df.to_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/antibody_tile_clustering/tile_clusters_pca_all_features.csv")



#features_np = np.load('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/syn1_all_features.npy', allow_pickle=True) 
#crop_names = np.load('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/pca_features/syn1_all_crop_paths.npy',  allow_pickle=True) 
#print(crop_names[0])
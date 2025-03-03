import os 
from PIL import Image
import random
import pandas as pd

output_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/tile_clustering/tile_clusters.csv"
output_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/tile_clustering/tile_clusters_pca.csv"
output_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/tile_clustering/tile_clusters_pca_all_features.csv"



output_path1 = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/tile_clustering/grouped_slides.csv"
output_path1 = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/tile_clustering/grouped_slides_pca.csv"
output_path1 = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/clustering_output/tile_clustering/grouped_slides_pca_all_features.csv"


df = pd.read_csv(output_path)
df["slide_name"] = df["filename"].apply(lambda l: l.split("/")[-2])
df["LBD_type"] = df["slide_name"].apply(lambda l: "PDD" if l.find("PD")!=-1 else "DLB")
print(df.head(2))
grouped_df = df.groupby(["LBD_type","slide_name","cluster"])["filename"].count()/df.groupby(["LBD_type","slide_name"])["filename"].count()
grouped_df =  grouped_df.reset_index()
grouped_df = grouped_df.sort_values(by = ["LBD_type","slide_name","filename"], ascending=False)
grouped_df.to_csv(output_path1)
#print(grouped_df)
#grouped_df = df.groupby(["LBD_type","cluster"])["filename"].count().reset_index()
#print(grouped_df.sort_values(by = ["LBD_type","filename"], ascending=False))
#print(grouped_df)
print(grouped_df.groupby(["LBD_type","cluster"])["filename"].mean())
print(grouped_df.groupby(["slide_name","cluster"])["filename"].mean())
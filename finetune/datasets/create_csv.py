from glob import glob
import os    
import pandas as pd
import numpy as np
    
dir_path =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/h5_files"

completed_slides =  glob(os.path.join(dir_path,"*.h5"))

h5_files = [ f.split("/")[-1].replace(".h5",".svs") for f in completed_slides]


print(len(h5_files))

print(h5_files[0])


df = pd.read_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/datasets/csv_files/valid_files_feb12.csv")

print(df.head(2))

print(len(df))

df1 =  df[df["filename"].isin(h5_files)]

print(len(df1))

print(df1.columns)

df2 = df1[["filename","patient_id","brain_region","anitibody_x","LBD_flag_x"]]

df2.columns =  ["slide_id","patient_id","brain_region","antibody","LBD_type"]

df2["label"] = np.where(df2["LBD_type"]=="PDD",0,1)

df2.to_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/data.csv")
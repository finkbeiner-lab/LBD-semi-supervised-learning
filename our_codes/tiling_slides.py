import sys
sys.path.append('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath')
print(sys.path)
import huggingface_hub
import gigapath
from gigapath.pipeline import tile_one_slide
import huggingface_hub
import os
import pandas as pd 
import gc
from glob import glob
import random
random.seed(4)

TILE_DIR = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/tiles"
#tile_one_slide(slide_path, save_dir=tmp_dir, level=1)
completed_slides =  glob(os.path.join(TILE_DIR,"*.svs"))
completed_slides = [x.split("/")[-1] for x in completed_slides]
#print(completed_slides)

#assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

#local_dir = os.path.join(os.path.expanduser("~"), ".cache/")
#local_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output"
#huggingface_hub.hf_hub_download("prov-gigapath/prov-gigapath", filename="sample_data/PROV-000-000001.ndpi", local_dir=local_dir, force_download=True)
#slide_path = os.path.join(local_dir, "sample_data/PROV-000-000001.ndpi")
#slide_path ="/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/PD13_Amygdala_C110-115 David Menassa.svs"

df = pd.read_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/datasets/csv_files/valid_files_feb12.csv")

#df = pd.read_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/datasets/csv_files/unique_patient_multi_antibodies.csv")

slide_paths =  df[df["LBD_flag_x"]=="PDD"]["filepath"].values
#print(slide_paths)

"""
slide_paths = [
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/PD133_C110-115_Amygdala David Menassa.svs",
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/PD258-Amyg-C34-45 David Menassa.svs",
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/PD131_C110-115_EntCx David Menassa.svs",
#"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/PD207-EntCx-C34-45 David Menassa.svs",
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/oxford_data2/DLB-2014-073-EntCx-Hippo-C110-115.svs",
#"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/oxford_data2/2015-007-C34-45-Striatum.svs",
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/oxford_data2/DLB-2013-131-Amygdala-C110-115.svs",
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/oxford_data2/2010-083-C34-45-Hippo.svs",
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/PD430-EntCx-C34-45 David Menassa.svs",
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/PD88_C110-115_EntCx David Menassa.svs",
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/PD334-Amyg-C34-45 David Menassa.svs",
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/oxford_data2/DLB-2015-005--EntCx-Hippo-C110-115.svs",
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/oxford_data2/DLB-2012-60-Amygdala-C110-115.svs",
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/PD171_C110-115_EntCx David Menassa.svs",
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/oxford_data2/DLB-1075-98-Striatum-C34-45.svs",
"/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/oxford_data2/2014-144-C34-45-Striatum.svs"
]
slide_paths = [
    "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/PD430-EntCx-C34-45 David Menassa.svs",
   "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/PD88_C110-115_EntCx David Menassa.svs"]
"""
#slide_paths = ["/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/oxford_data2/PD430-EntCx-C34-45 David Menassa.svs"]

#save_dir = os.path.join(local_dir, 'tiles/')
random.shuffle(slide_paths)
c = 0
print("NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides. Please make sure to use the appropriate level for the 0.5 MPP")
for slide in slide_paths:
    print("Count: ", c)
    if slide.split("/")[-1] not in completed_slides:
        print("processing slide ", slide)
        try:
            tile_one_slide(slide, save_dir=TILE_DIR, level=0, tile_size=512) # old 1
            torch.cuda.empty_cache()  # Free unused memory
            #c = c+1
            #if c>5:
            #    break
        except Exception as e:
            print(e)
    else:
        print("already completed", slide)
    

    #break

print("NOTE: tiling dependency libraries can be tricky to set up. Please double check the generated tile images.")

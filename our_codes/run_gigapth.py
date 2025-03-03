import os
import sys
sys.path.append('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath')
print(sys.path)
import huggingface_hub
import gigapath
print(dir(gigapath))
from gigapath.pipeline import tile_one_slide
from gigapath.pipeline import load_tile_slide_encoder
from gigapath.pipeline import run_inference_with_tile_encoder
from gigapath.pipeline import run_inference_with_slide_encoder
from glob import glob
import numpy as np

os.environ["HF_TOKEN"] = "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"

assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

#local_dir = os.path.join(os.path.expanduser("~"), ".cache/")
#huggingface_hub.hf_hub_download("prov-gigapath/prov-gigapath", filename="sample_data/PROV-000-000001.ndpi", local_dir=local_dir, force_download=False)
#slide_path = os.path.join(local_dir, "sample_data/PROV-000-000001.ndpi")

def extract_encoding_one_slide(slide_dir):
    image_paths = [os.path.join(slide_dir, img) for img in os.listdir(slide_dir) if img.endswith('.png')]
    print(f"Found {len(image_paths)} image tiles")
    if len(image_paths)>0:
        tile_encoder, slide_encoder_model = load_tile_slide_encoder(global_pool=True)
        tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)

        for k in tile_encoder_outputs.keys():
            print(f"tile_encoder_outputs[{k}].shape: {tile_encoder_outputs[k].shape}")
        slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=slide_encoder_model, **tile_encoder_outputs)
        print(slide_embeds.keys())
        np.save('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/slide_encoding/'+slide_dir.split("/")[-1]+'.npy', slide_embeds) 
    else:
        print("no tiles found for ", slide_dir)




TILE_DIR = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/tiles/output/"
#tile_one_slide(slide_path, save_dir=tmp_dir, level=1
TILE_DIR = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/gigapath_syn1_crops"


slide_paths =  glob(os.path.join(TILE_DIR,"*.svs"))
print(slide_paths)
SLIDE_ENCODING_DIR = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/output/slide_encoding'
completed_slides = glob(os.path.join(SLIDE_ENCODING_DIR,"*.npy"))
completed_slides = [x.split("/")[-1] for x in completed_slides]
 

for slide_dir in slide_paths:
    if slide_dir.split("/")[-1]+".npy" not in  completed_slides:
        print("processing slide", slide_dir.split("/")[-1]+".npy")
        #break
        extract_encoding_one_slide(slide_dir)
    else:
        print("completed slide", slide_dir.split("/")[-1]+".npy" )


import huggingface_hub
import os
import sys
sys.path.append('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath')
import gigapath
from gigapath.preprocessing.data.slide_utils import find_level_for_target_mpp

#assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

local_dir = os.path.join(os.path.expanduser("~"), ".cache/")
#huggingface_hub.hf_hub_download("prov-gigapath/prov-gigapath", filename="sample_data/PROV-000-000001.ndpi", local_dir=local_dir, force_download=True)
#print(local_dir)
#slide_path = os.path.join(local_dir, )
#slide_path ="/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/PD13_Amygdala_C110-115 David Menassa.svs"
slide_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_data/oxford_data2/2008-134-C34-45-Amygdala.svs"

print("NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides")
target_mpp = 0.33
level = find_level_for_target_mpp(slide_path, target_mpp)
if level is not None:
    print(f"Found level: {level}")
else:
    print("No suitable level found.")

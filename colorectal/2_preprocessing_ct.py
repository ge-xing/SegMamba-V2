
from light_training.preprocessing.preprocessors.preprocessor_coloretal import DefaultPreprocessor 
import numpy as np 
import pickle 
import json 


base_dir = "/data/xingzhaohu/colorectal/data/raw_data/random2500"
image_dir = "Image"
label_dir = "ROI"

def process_train():

    preprocessor = DefaultPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    label_dir=label_dir,
                                   )

    out_spacing = [3.75, 0.79296899, 0.79296899]
    output_dir = "./data/fullres/train/"
    
    with open("./data_analysis_result.txt", "r") as f:
        content = f.read().strip("\n")
        print(content)
    content = json.loads(content)
    foreground_intensity_properties_per_channel = content["intensity_statistics_per_channel"]
    
    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir, 
                     all_labels=[1,],
                     foreground_intensity_properties_per_channel=foreground_intensity_properties_per_channel)
    
def plan():
    preprocessor = DefaultPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    label_dir=label_dir
                                   )
    
    preprocessor.run_plan()

if __name__ == "__main__":
# 
    plan()
    process_train()

    

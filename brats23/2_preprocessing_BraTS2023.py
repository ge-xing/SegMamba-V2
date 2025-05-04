
from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor 
import numpy as np 
import pickle 
import json 

data_filename = ["t2w.nii.gz",
                 "t2f.nii.gz",
                 "t1n.nii.gz",
                 "t1c.nii.gz"]
seg_filename = "seg.nii.gz"

def process_train():
    base_dir = "./data/raw_data/"
    image_dir = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )

    out_spacing = [1.0, 1.0, 1.0]
    output_dir = "./data/fullres/train/"
    
    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir, 
                     all_labels=[1, 2, 3],
    )

def plan():
    base_dir = "./data/raw_data/"
    image_dir = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )
    
    preprocessor.run_plan()


if __name__ == "__main__":
# 
    plan()
    process_train()
  

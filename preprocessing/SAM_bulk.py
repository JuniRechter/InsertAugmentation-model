# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:11:23 2024

@author: Juniper Rechter
"""

import os
import sys
import ast
import numpy as np
import pandas as pd
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
import argparse
import time
#import humanfriendly

#%%
'''
This function ensures that our bounding boxes are in the correct format, otherwise they are returned as strings.
'''
def literal_return(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError) as e:
        return val

#%%
'''
MegaDetector notes:
    MD batch detections produces bounding boxes in the form of [x_min, y_min, width_of_box, height_of_box],
    where (x_min, y_min) is the upper-left corner of the box. Both these coordinates and the box width and height are 
    relative to the size of image.
    ie. If an image maintains its aspect ratio, these coordinates will apply. 
    However, because our resized images have had padding to make them square, these bboxes will have to be applied to the original 
    image, and the cropped animal can be resized for insertion later. 

Bbox: [[x_min, y_min, width_of_box, height_of_box]]
Therefore, input_box is:
    [(x_min * W), (y_min * H), ((x_min * W) + (width_of_box * W)), ((y_min * H) + (height_of_box * H))]
    or
    [(bbox[0]*W), (bbox[1] * H), ((bbox[0] * W) + (bbox[2] * W)), ((bbox[1] * H) + (bbox[3] * H))]
'''
#%%
'''
SAM notes:
    predict(self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]

        box (np.ndarray or None): A length 4 array given a box prompt 
                                  to the model, in XYXY format.
'''
#%%
'''
Set up the SAM model and predictor.
'''
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
#%%
'''
Grab our image and set the image height and width for later cropping and adjusting mask dimensions.
'''
def gimme_animal(df,
                 CT="CT",
                 output_directory="output_directory"):

    df = pd.read_csv(df)
    df['bbox'] = df['bbox'].apply(literal_return)
    df = df.dropna(subset=['bbox'])
    df = df.dropna(subset=['species'])

    for i in df.index:
        image = cv2.imread(df['path'][i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        path = df['path'][i]
        if CT=="AHC":
            path = path.split('/')[3]
        elif CT=="MNRF":
            root, year, area, filename = path.split('/')
            path = area + '~' + filename
        path = path[:-4]
        spp = df['species'][i]
        H = image.shape[0] #set the image height
        W = image.shape[1] #set the image width
        bbox = df['bbox'][i] #grab bbox values
    #Note:  the int function here will generally round down, even if the float is 1.97, etc.
        input_box = [int(bbox[0]*W),                        #x1
                     int(bbox[1] * H),                      #y1 
                     int((bbox[0] * W) + (bbox[2] * W)),    #x2
                     int((bbox[1] * H) + (bbox[3] * H))]    #y2
        predictor.set_image(image)
        input_box = np.array(input_box)

        masks, scores, logits = predictor.predict(point_coords=None,
                                                  point_labels=None,
                                                  box=input_box[None, :],
                                                  multimask_output=True,
                                                  return_logits=False)
        mask = np.reshape(masks[1], (H, W))
        mask=mask.astype(np.uint8)
        mask_image = (mask *255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        image[:, :, 3] = mask_image
        y1=input_box[1]
        y2=input_box[3]
        x1=input_box[0]
        x2=input_box[2]
        cropped_image = image[y1:y2, x1:x2]
        if not os.path.exists(output_directory + "/cropped/" + spp):
            os.makedirs(output_directory + "/cropped/" + spp)
        cv2.imwrite(output_directory + "/cropped/" + spp + "/" + path + "_" + spp + "_mask" + ".png", cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGBA))
        if (i+1)%50==0:
            print(((df['path'][i]) + " segmented and saved. \n Image {} of {}.").format((i+1), len(df)), flush=True)

#%% Command-line driver

def main():

    parser = argparse.ArgumentParser(
        description='Program to crop info-bars from CT images and resize in preparation for ML input')
    parser.add_argument('df', 
                        type=str,
                        help='Path to directory of images.')
    parser.add_argument('output_directory', 
                        type=str,
                        help='Path to the directory to save resized images.')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.df), \
        'df {} does not exist'.format(args.df)            
    assert os.path.exists(args.output_directory), \
        'Output directory {} does not exist'.format(args.output_directory)

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    print("SamPredictor initiated. Running segmentation.", flush=True)

    start_time = time.time()

    gimme_animal(df=args.df,
                 output_directory=args.output_directory)

    elapsed = time.time() - start_time

    print("I'm finished! Finally, I'm a beautiful butterfly!")
    print(('Finished segmenting images in {:.3f} seconds.').format(elapsed))
#    print(('Finished segmenting images in {}.').format(humanfriendly.format_timespan(elapsed)))

if __name__ == '__main__':
    main()

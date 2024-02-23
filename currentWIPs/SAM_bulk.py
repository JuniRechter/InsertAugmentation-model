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
import matplotlib.pyplot as plt
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
def get_animal(df):
    df = pd.read_csv(df)
    df['bbox'] = df['bbox'].apply(literal_return)
    
    #Note:  the int function here will generally round down, even if the float is 1.97, etc.
    for i in df['path']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H = image.shape[0] #set the image height
        W = image.shape[1] #set the image width
        bbox = df['bbox'][i] #grab bbox values
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
        masks.shape  # (number_of_masks) x H x W
        for m in masks:
            mask=np.reshape(masks[m], (H, W)) #change the mask shape to remove the first channel, but keep og dimensions
            #Masks are output as True, False, binaries for every pixel. Uint8 format changes that to 1s and 0s.
            mask=mask.astype(np.uint8)
            mask_image = (mask * 255).astype(np.uint8) #This mask_image output is the one we want.
            cv2.imwrite('mask.png', mask_image)
            
#            image = cv2.imread(i)
#            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Read into the BGR format cos CV2 is weird about RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA) #add alpha layer to the image
            image[:, :, 3] = mask_image #Apply mask to the alpha layer
            cv2.imwrite('masked_image3.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)) #Convert back to the correct form for saving. 
    
            #image crop takes the form y1:y2, x1:x2 #Yeah I don't know, bboxes are dumb
            y1=input_box[1]
            y2=input_box[3]
            x1=input_box[0]
            x2=input_box[2]
            cropped_image = image[y1:y2, x1:x2] #apply MD box here to reduce the amount of empty space
            cv2.imwrite('cropped_mask_image.png', cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGBA))
#%%
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2)) 
#%%
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_box(input_box, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()
#%%
'''
Here we convert the mask from a binary format (where every pixel is represented by either 'True' or 'False'), 
to a readable uint8 format (where pixels are represented by '1' or '0'). 
We then multiply these values by 255, the maximum value in an RGB pixel channel, where 255 is a full/True pixel, and 
0 is an empty/False pixel.
This format is readable by CV2 as a black and white mask which can be applied to other images. 
'''
#Constructing readable mask for cv2
mask=np.reshape(masks[1], (H, W)) #change the mask shape to remove the first channel, but keep og dimensions
#Masks are output as True, False, binaries for every pixel. Uint8 format changes that to 1s and 0s.
mask=mask.astype(np.uint8)
mask_image = (mask * 255).astype(np.uint8) #This mask_image output is the one we want.
cv2.imwrite('mask.png', mask_image)
#%%
'''
If we apply the original uint8 mask to an image, it will keep the black background, which can make things harder 
for inserting animals later. Nevertheless, this clarifies what the final image will include.
'''
#Constructing image with positive mask elements, but background remains black
output = cv2.bitwise_and(image, image, mask = mask)
cv2.imwrite('masked_image2.png', cv2.cvtColor(output, cv2.COLOR_RGB2BGR)) 
#%%
'''
Add an alpha layer to the original image. The alpha layer determines the opacity or transparency of each pixel.
Even if the other 3 channels create "blue", an alpha value of 0 means this pixel is transparent.
We apply the black and white mask to this alpha layer. Black changes the alpha to 0, white to 1.
This removes opacity in areas where the mask is black, therefore leaving only the animal mask we want with 
with a transparent background. 
'''
image = cv2.imread("original/03290020.JPG")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Read into the BGR format cos CV2 is weird about RGB
image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA) #add alpha layer to the image
image[:, :, 3] = mask_image #Apply mask to the alpha layer
cv2.imwrite('masked_image3.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)) #Convert back to the correct form for saving. 
#%%
cropped_image = image[1900:2800, 800:1900] #apply MD box here to reduce the amount of empty space
plt.imshow(cropped_image)
cv2.imwrite('cropped_mask_image.png', cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGBA))

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:29:07 2024

@author: Juniper Rechter

Planned pseudo-code for inserting animal mask crops into empty CT images.
"""

import os
import sys
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from skimage.exposure import match_histograms
import cv2
import argparse
import time
import humanfriendly
#%%
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
#%%
mask_augs = iaa.Sequential([iaa.Resize((0.09, 0.17), interpolation="area"),
        iaa.SomeOf((1, 2), [  # Random number between 0, 2
        iaa.Fliplr(0.5),  # Horizontal flips                     0.01
        iaa.Rotate((-10, 10)),
        ], random_order=True)], random_order = True)

#%%
'''
Function from StackOverflow user Ben: https://stackoverflow.com/users/59850/ben
https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
'''
def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: 
        x_offset = (bg_w - fg_w) // 2
    else:
        centre_x = (bg_w - fg_w) // 2
        x_offset = x_offset + centre_x
        
    if y_offset is None: 
        y_offset = (bg_h - fg_h) // 2
    else:
        centre_y = (bg_h - fg_h) // 2
        y_offset = y_offset + centre_y

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
#    foreground_colors = match_histograms(foreground_colors, background, channel_axis=-1)
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

#%%
output_directory="small"
#Getting OG background image
composite = []
for i in df.index:
    path = df['filename'][i]
    subdir = path.rsplit('/', 1)[0]
    night = df['Night'][i]
    if night==True:
        image = cv2.imread("resized/" + path)
        crop_index = crops[crops.Night==True].sample(n=1, replace=False).index
        crop_index = crop_index[0]
    elif night==False:
        image_og = cv2.imread("resized/" + path)
        image = cv2.imread("resized/" + path)
        crop_index = crops[crops.Night==False].sample(n=1, replace=False).index
        crop_index = crop_index[0]
    crop = cv2.imread('original/AHC/crops/' + crops['filename'][crop_index], cv2.IMREAD_UNCHANGED)
    spp = crops['species'][crop_index]
    crop_aug = mask_augs.augment_image(crop)

    y_jitter=np.random.randint(-7, 40)
    x_jitter=np.random.randint(-85, 85)
    add_transparent_image(image, crop_aug, x_offset=x_jitter, y_offset=y_jitter)
    if night ==True:
        noise = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=(0, 5))])
        image = noise.augment_image(image)
    if night==False:
        image = match_histograms(image, image_og, channel_axis=-1)
    image = cv2.rectangle(image, (0,0), (244,52), (0,0,0), -1) # 52 pixels on top and bottom Black infobar)
    image = cv2.rectangle(image, (0,172), (244,244), (0,0,0), -1) # 52 pixels on top and bottom Black infobar)
    composite.append({'bg_filename':df['filename'][i],
                      'crop_filename':crops['filename'][crop_index],
                      'species':spp,
                      'Night':night})
    if not os.path.exists(output_directory + "/" + subdir):
        os.makedirs(output_directory + "/" + subdir)
    cv2.imwrite(output_directory + "/" + path, image)
    rolld20=np.random.randint(1, 21)
    if rolld20==20:
        if not os.path.exists("critical/" + subdir):
            os.makedirs("critical/" + subdir)
        cv2.imwrite("critical/" + path, image)
comp_df = pd.DataFrame(composite)
comp_df.to_csv("compositedf2.csv", index=False)

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:46:22 2024

@author: Juniper Rechter

"""
#%%
import tensorflow as tf
import cv2
from imgaug import augmenters as iaa
from skimage.exposure import match_histograms

import collections
collections.Iterable = collections.abc.Iterable
import numpy as np
np.bool = np.bool_
np.complex = np.complex_

import argparse
import os
import sys
import time
#import humanfriendly
#%% Define mask insertion function
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
#%% Define augmentation functions

mask_augs = iaa.Sequential([iaa.Resize((0.09, 0.17), interpolation="area"),
        iaa.SomeOf((1, 2), [  # Random number between 0, 2
        iaa.Fliplr(0.5),  # Horizontal flips                     0.01
        iaa.Rotate((-10, 10)),
        ], random_order=True)], random_order = True)

Winter_aug = iaa.SomeOf((1, 3), [  # Random number between 0, 3
        iaa.weather.Clouds(),
        iaa.weather.Fog(),
        iaa.weather.Snowflakes()
        ], random_order=True)

small_aug = iaa.SomeOf((1, 3), [  # Random number between 0, 3
        iaa.Fliplr(0.5),  # Horizontal flips                     0.01
        iaa.Multiply((0.95, 1.05), per_channel=0.2),  # Brightness multiplier per channel    0.05
        iaa.Grayscale(alpha=(0.1, 1.0)),  # Random Grayscale conversion          0.17        
        iaa.LinearContrast((0.75, 1.5)),  # Contrast Normalization               0.95
        ], random_order=True)

train_aug = iaa.SomeOf((1, 3), [  # Random number between 0, 3
        iaa.Fliplr(0.5),  # Horizontal flips                     0.01
        # Random channel increase and rotation 0.03
        iaa.Add((-5, 5)),  # Overall Brightness                   0.04
        iaa.Multiply((0.95, 1.05), per_channel=0.2),  # Brightness multiplier per channel    0.05
        iaa.Sharpen(alpha=(0.1, 0.75), lightness=(0.85, 1.15)),  # Sharpness                            0.05
        iaa.WithColorspace(to_colorspace='HSV', from_colorspace='RGB',  # Random HSV increase                  0.09
                           children=iaa.WithChannels(0, iaa.Add((-30, 30)))),
        iaa.WithColorspace(to_colorspace='HSV', from_colorspace='RGB',
                           children=iaa.WithChannels(1, iaa.Add((-30, 30)))),
        iaa.WithColorspace(to_colorspace='HSV', from_colorspace='RGB',
                           children=iaa.WithChannels(2, iaa.Add((-30, 30)))),
        iaa.AddElementwise((-10, 10)),  # Per pixel addition                   0.11
        iaa.CoarseDropout((0.0, 0.02), size_percent=(0.02, 0.25)),  # Add large black squares              0.13
        iaa.GaussianBlur(sigma=(0.1, 1.0)),  # GaussianBlur                         0.14
        iaa.Grayscale(alpha=(0.1, 1.0)),  # Random Grayscale conversion          0.17
        iaa.Dropout(p=(0, 0.1), per_channel=0.2),  # Add small black squares              0.17
        iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Add Gaussian per pixel noise         0.26
        iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.25),  # Distort image by rearranging pixels  0.70
        iaa.LinearContrast((0.75, 1.5)),  # Contrast Normalization               0.95
        iaa.weather.Clouds(),
        iaa.weather.Fog(),
  #      iaa.weather.Rain()
    ], random_order=True)

#%%

class TrainingDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, crops, X_col, y_col,
                 batch_size, model_name,
                 shuffle=True, inserts=False):
        
        self.df = df.copy()
        self.crops = crops.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.model = model_name
        self.inserts=inserts
        self.n = len(self.df)
        if self.inserts==False:
            self.n_id = ((df[y_col['id']]).nunique())
        elif self.inserts==True:
            self.n_id = ((df[y_col['id']]).nunique()-1)
        self.n_domain = df[y_col['domain']].nunique()
        
    # Shuffling upon epoch end if flagged
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __insert_augs(self, path, night):
        if night==True:
            image = cv2.imread(path)
            crop_index = self.crops[self.crops.Night==True].sample(n=1, replace=False).index
            crop_index = crop_index[0]
        elif night==False:
            image_og = cv2.imread(path)
            image = cv2.imread(path)
            crop_index = self.crops[self.Night==False].sample(n=1, replace=False).index
            crop_index = crop_index[0]
        crop = cv2.imread('crops/' + self.crops['filename'][crop_index], cv2.IMREAD_UNCHANGED)
        species = self.crops['species'][crop_index]
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

        rolld20=np.random.randint(1, 21)
        if rolld20==20: #Save 5% of created images for verification
            subdir = path.rsplit('/', 1)[0]
            if not os.path.exists("Results/augmentations/" + subdir):
                os.makedirs("Results/augmentations/" + subdir)
            cv2.imwrite("Results/augmentations/" + path, image)
        image_arr = image.astype("float32")
        return image_arr, species

    #Reads image and normalizes it
    def __get_input(self, x, inserts):
        if inserts==True:
            path=x[0]
            empty=x[1]
            night=x[2]

            if empty==True:
                image_arr, x[3] = self.__insert_augs(path, night)
            else:
                image = tf.keras.utils.load_img(path)
                image_arr = tf.keras.utils.img_to_array(image)
            image_arr = small_aug.augment_image(image_arr.astype("uint8")).astype("float32")
            return image_arr/255, x[3]
        elif inserts==False:
            image = tf.keras.utils.load_img(path)
            image_arr = tf.keras.utils.img_to_array(image)
            image_arr = small_aug.augment_image(image_arr.astype("uint8")).astype("float32")
            return image_arr/255

    #One-hot encoding of label
    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)

    #Generates batch_size samples of data
    def __get_data(self, batches):
        # Generates data containing batch_size samples
        path_batch = batches[self.X_col['path']]
        name_batch = batches[self.y_col['id']]
        
        if self.inserts==True:
            empty_batch = batches[self.X_col['empty']]
            night_batch = batches[self.X_col['Night']]
            x_batches=np.array([path_batch, empty_batch, night_batch, name_batch])
            x_batches=np.stack(x_batches,axis=1)
            X_batch, y0_batch = np.asarray([self.__get_input(x, True) for x in x_batches])
        else:
            X_batch = np.asarray([self.__get_input(x, False) for x in path_batch])
            y0_batch = np.asarray([self.__get_output(y, self.n_id) for y in name_batch])

        if self.model in ('DenseNet201', 'CNN'):
            return X_batch, y0_batch
        elif self.model in ('DANNseNet201', 'DANN'):
            sin_season_batch = batches[self.y_col['sin_date']]
            cos_season_batch = batches[self.y_col['cos_date']]
            y1_batch = np.asarray([y for y in zip(sin_season_batch, cos_season_batch)])
            return X_batch, tuple([y0_batch, y1_batch])
        elif self.model in ('CatDANN', 'catDANN'):
            domain_batch = batches[self.y_col['domain']]
            y1_batch = np.asarray([self.__get_output(y, 4) for y in domain_batch])
            return X_batch, tuple([y0_batch, y1_batch])

    #Returns data batches as tuple
    def __getitem__(self, index):

        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)    

        return X, y

    #Returns number of batches
    def __len__(self):
        return self.n // self.batch_size

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
import random
import collections
collections.Iterable = collections.abc.Iterable
import numpy as np
np.bool = np.bool_
np.complex = np.complex_

import argparse
import os
import sys
import time
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
    return background

#%% Define augmentation functions
mask_augs = iaa.Sequential([iaa.Resize((0.45, 0.85), interpolation="area"), #area interpolation is best for shrinking images
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
        iaa.LinearContrast((0.75, 1.5)),  # Contrast Normalization               0.95
        ], random_order=True)

train_aug = iaa.SomeOf((1, 3), [  # Random number between 0, 3
        iaa.Fliplr(0.5),  # Horizontal flips                     0.01
        iaa.Add((-5, 5)),  # Overall Brightness                   0.04
        iaa.Multiply((0.95, 1.05), per_channel=0.2),  # Brightness multiplier per channel    0.05
        iaa.Sharpen(alpha=(0.1, 0.75), lightness=(0.75, 1.1)),  # Sharpness 0.05
        iaa.GammaContrast((0.5, 2.0)),
        iaa.WithHueAndSaturation([iaa.WithChannels(0, iaa.Add((-15,15))), 
                                  iaa.WithChannels(1, [iaa.Multiply((0.5, 1.5)), iaa.LinearContrast((0.75, 1.25))])]),
        iaa.AddElementwise((-10, 10), per_channel=0.5),  # Per pixel addition                   0.11
        iaa.CoarseDropout((0.0, 0.02), size_percent=(0.02, 0.25)),  # Add large black squares              0.13
        iaa.GaussianBlur(sigma=(0.1, 0.5)),  # GaussianBlur                         0.14
        iaa.Dropout(p=(0, 0.1), per_channel=0.2),  # Add small black squares              0.17
        iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255), per_channel=0.5),  # Add Gaussian per pixel noise         0.26
#        iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.25),  # Distort image by rearranging pixels  0.70
        iaa.LinearContrast((0.75, 1.5)),  # Contrast Normalization               0.95
        iaa.weather.Clouds(),
    ], random_order=True)

nightnoise = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=(0, 3))])
#%%
class TrainingDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, crops, X_col, y_col,
                 batch_size, model_name, shuffle=True, 
                 inclEmpties=True, INSERTS=False, SPP_BAL=False,
                 weights=None, night_weights=None):

        self.df = df.copy()
        self.crops = crops.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.model = model_name
        self._inclEmpties = inclEmpties
        self.INSERTS=INSERTS
        self.n = len(self.df)
        
        if self._inclEmpties==True:
            day_classes = [0, 1, 2, 3, 4, 5, 6, 7]
            night_classes = [0, 1, 2, 3, 4, 6, 7]
        else:
            day_classes = [0, 1, 2, 3, 4, 5, 6]
            night_classes = [0, 1, 2, 3, 5, 6]
        if self._inclEmpties==False:
            self.n_id = df[y_col['id']].nunique()
        elif self.INSERTS==True:
            self._weights=weights
            self._night_weights=night_weights
            self.n_id = 7
            if self.df['id'].isin([-2]).any():
                if SPP_BAL == False:
                    self.NL_weights=weights
                    self.NL_night_weights=night_weights
                else:
                    self.NL_weights=None
                    self.NL_night_weights=None
            if self.n_id>12:
                day_choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 25, 28, 29, 30, 31, 32, 33, 34, 35]
                night_choices=[0, 1, 2, 4, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33]
        self.n_domain = df[y_col['domain']].nunique()


    # Shuffling upon epoch end if flagged
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __insert_augs(self, path, night, spp):
        if spp==-1:
            if night==False:
                rolld8=(random.choices(day_classes, weights=self._weights, k=1))[0]
                if self._inclEmpties==True:
                    if rolld8==0:
                        image = tf.keras.utils.load_img(path)
                        image_arr = tf.keras.utils.img_to_array(image)
                        species=0
                        return image_arr, species

                crop_index = self.crops.loc[(self.crops.Night==False) & (self.crops.id==rolld8)].sample(n=1).index.item()
                cpath = self.crops.iloc[crop_index, 0]
                crop = cv2.imread(('crops/' + cpath), cv2.IMREAD_UNCHANGED)
                spp_name = self.crops.iloc[crop_index, 2]
                species = self.crops.iloc[crop_index, 3]
                
                alpha = crop[:,:,3]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
                reference = cv2.imread(path)
                reference = reference[52:172, 0:224]

                crop2 = match_histograms(crop, reference, channel_axis = -1)
                crop = cv2.addWeighted(crop2, 0.25, crop, 0.75, 0.0)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
                crop[:,:,3] = alpha
                crop_aug = mask_augs.augment_image(crop)
                y_jitter=random.randint(-7, 40)
                x_jitter=random.randint(-85, 85)
                image = cv2.imread(path)
                image = add_transparent_image(image, crop_aug, x_offset=x_jitter, y_offset=y_jitter)

                image_og = cv2.imread(path)
         #       image = match_histograms(image, image_og, channel_axis=-1)
            elif night==True:
                nightd7=(random.choices(night_classes, weights=self._night_weights, k=1))[0]
                if self._inclEmpties==True:
                    if nightd7==0:
                        image = tf.keras.utils.load_img(path)
                        image_arr = tf.keras.utils.img_to_array(image)
                        species=0
                        return image_arr, species

                crop_index = self.crops.loc[(self.crops.Night==True) & (self.crops.id==nightd7)].sample(n=1).index.item()
                cpath = self.crops.iloc[crop_index, 0]
                crop = cv2.imread(('crops/' + cpath), cv2.IMREAD_UNCHANGED)
                spp_name = self.crops.iloc[crop_index, 2]
                species = self.crops.iloc[crop_index, 3]

                crop_aug = mask_augs.augment_image(crop)
                y_jitter=random.randint(-7, 40)
                x_jitter=random.randint(-85, 85)
                image = cv2.imread(path)
                image = add_transparent_image(image, crop_aug, x_offset=x_jitter, y_offset=y_jitter)
                image = nightnoise.augment_image(image)

            image = cv2.rectangle(image, (0,0), (244,52), (0,0,0), -1) # 52 pixels on top and bottom Black infobar)
            image = cv2.rectangle(image, (0,172), (244,244), (0,0,0), -1) # 52 pixels on top and bottom Black infobar)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rolld1000=random.randint(1, 1001)
            if rolld1000==1000: #Save 1% of created images for verification
                file = path.rsplit('/')[-1]
                spp_name = self.crops.iloc[crop_index, 2]
                if not os.path.exists("Results/augmentations/"):
                    os.makedirs("Results/augmentations/")
                cv2.imwrite("Results/augmentations/" + spp_name + "_" + file, image)

            image_arr = image.astype("float32")
            species = self.crops.iloc[crop_index, 3]
            return image_arr, species
        if spp==-2:
            if night==False:
                rolld8=(random.choices(day_classes, weights=self.NL_weights, k=1))[0]
                if self._inclEmpties==True:
                    if rolld8==0:
                        image = tf.keras.utils.load_img(path)
                        image_arr = tf.keras.utils.img_to_array(image)
                        species=0
                        return image_arr, species

                crop_index = self.crops.loc[(self.crops.Night==False) & (self.crops.id==rolld8)].sample(n=1).index.item()
                cpath = self.crops.iloc[crop_index, 0]
                crop = cv2.imread(('crops/' + cpath), cv2.IMREAD_UNCHANGED)
                spp_name = self.crops.iloc[crop_index, 2]
                species = self.crops.iloc[crop_index, 3]
                crop_aug = mask_augs.augment_image(crop)
                y_jitter=random.randint(-7, 40)
                x_jitter=random.randint(-85, 85)
                image = cv2.imread(path)
                image = add_transparent_image(image, crop_aug, x_offset=x_jitter, y_offset=y_jitter)

                image_og = cv2.imread(path)
                image = insert_aug.augment_image(image)
                #image = match_histograms(image, image_og, channel_axis=-1)
            elif night==True:
                nightd7=(random.choices(night_classes, weights=self.NL_night_weights, k=1))[0]
                if self._inclEmpties==True:
                    if nightd7==0:
                        image = tf.keras.utils.load_img(path)
                        image_arr = tf.keras.utils.img_to_array(image)
                        species=0
                        return image_arr, species

                crop_index = self.crops.loc[(self.crops.Night==True) & (self.crops.id==nightd7)].sample(n=1).index.item()
                cpath = self.crops.iloc[crop_index, 0]
                crop = cv2.imread(('crops/' + cpath), cv2.IMREAD_UNCHANGED)
                spp_name = self.crops.iloc[crop_index, 2]
                species = self.crops.iloc[crop_index, 3]

                crop_aug = mask_augs.augment_image(crop)
                y_jitter=random.randint(-7, 40)
                x_jitter=random.randint(-85, 85)
                image = cv2.imread(path)
                image = add_transparent_image(image, crop_aug, x_offset=x_jitter, y_offset=y_jitter)
                image = nightnoise.augment_image(image)

            image = cv2.rectangle(image, (0,0), (244,52), (0,0,0), -1) # 52 pixels on top and bottom Black infobar)
            image = cv2.rectangle(image, (0,172), (244,244), (0,0,0), -1) # 52 pixels on top and bottom Black infobar)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            rolld1000=random.randint(1, 1001)
            if rolld1000==1000: #Save 1% of created images for verification
                file = path.rsplit('/')[-1]
                spp_name = self.crops.iloc[crop_index, 2]
                if not os.path.exists("Results/augmentations/"):
                    os.makedirs("Results/augmentations/")
                cv2.imwrite("Results/augmentations/" + spp_name + "_" + file, image)

            image_arr = image.astype("float32")
            species = self.crops.iloc[crop_index, 3]
            return image_arr, species
#%%

class ValidationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, X_col, y_col,
                 batch_size, model_name,
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.model = model_name
        self.n = len(self.df)
        self.n_id = ((df[y_col['id']]).nunique())
        self.n_domain = df[y_col['domain']].nunique()
    # Shuffling upon epoch end if flagged
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    #Reads image and normalizes it
    def __get_input(self, path):

        image = tf.keras.utils.load_img(path)
        image_arr = tf.keras.utils.img_to_array(image)
        return image_arr/255.
   
    #One-hot encoding of label
    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)
    
    #Generates batches of data
    def __get_data(self, batches):
        # Generates data containing batch_size samples
        path_batch = batches[self.X_col['path']]
        name_batch = batches[self.y_col['id']]
        X_batch = np.asarray([self.__get_input(x) for x in path_batch])
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


#%%
class TestingDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, X_col, y_col,
                 batch_size, model_name,
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.model = model_name
        self.n = len(self.df)
        self.n_id = ((df[y_col['id']]).nunique())
        self.n_domain = 4

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, path):

        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        return image_arr/255.
    
    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples
        path_batch = batches[self.X_col['path']]
        name_batch = batches[self.y_col['id']]
        X_batch = np.asarray([self.__get_input(x) for x in path_batch])
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
        
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)    
        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size

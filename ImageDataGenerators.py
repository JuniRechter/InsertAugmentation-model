# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:57:27 2023

@author: Juniper Rechter

"""
#%%
import tensorflow as tf
from imgaug import augmenters as iaa
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
#%% Define augmentation functions

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
    def __init__(self, df, X_col, y_col,
                 batch_size,
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.n = len(self.df)
        self.n_id = ((df[y_col['id']]).nunique())
        self.n_domain = df[y_col['domain']].nunique()
    
    # Shuffling upon epoch end if flagged
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    #Reads image and normalizes it
    def __get_input(self, path):
        image = tf.keras.utils.load_img("MNRF/" + path)
        image_arr = tf.keras.utils.img_to_array(image)
#        image_arr = small_aug.augment_image(image_arr.astype("uint8")).astype("float32")
        return image_arr/255.
    
    #One-hot encoding of label
    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)

    #Generates batch_size samples of data
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path_batch = batches[self.X_col['path']]
        name_batch = batches[self.y_col['id']]
        
        sin_season_batch = batches[self.y_col['sin_date']]
        cos_season_batch = batches[self.y_col['cos_date']]
#        domain_batch = batches[self.y_col['domain']]

        X_batch = np.asarray([self.__get_input(x) for x in path_batch])
        y0_batch = np.asarray([self.__get_output(y, self.n_id) for y in name_batch])
        y1_batch = np.asarray([y for y in zip(sin_season_batch, cos_season_batch)])
#        y1_batch = np.asarray([self.__get_input(y) for y in domain_batch])

        #return X_batch, y0_batch
        return X_batch, y0_batch   
#        return X_batch, tuple([y0_batch, y1_batch])
    
    #Returns data batches as tuple
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)    
        
        return X, y
    
    #Returns number of batches
    def __len__(self):
        return self.n // self.batch_size
#%%

class ValidationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, X_col, y_col,
                 batch_size,
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.n = len(self.df)
        self.n_id = ((df[y_col['id']]).nunique())
        self.n_domain = df[y_col['domain']].nunique()
    
    # Shuffling upon epoch end if flagged
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    #Reads image and normalizes it
    def __get_input(self, path):

        image = tf.keras.utils.load_img("AHC/" + path)
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
        
        sin_season_batch = batches[self.y_col['sin_date']]
        cos_season_batch = batches[self.y_col['cos_date']]
#        domain_batch = batches[self.y_col['domain']]

        X_batch = np.asarray([self.__get_input(x) for x in path_batch])
        y0_batch = np.asarray([self.__get_output(y, self.n_id) for y in name_batch])
        y1_batch = np.asarray([y for y in zip(sin_season_batch, cos_season_batch)])
#        y1_batch = np.asarray([self.__get_input(y) for y in domain_batch])

        #return X_batch, y0_batch
        return X_batch, y0_batch   
#        return X_batch, tuple([y0_batch, y1_batch])
    
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
                 batch_size,
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.n = len(self.df)
        self.n_id = ((df[y_col['id']]).nunique())
        self.n_domain = 4
    
        
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, path):

        image = tf.keras.preprocessing.image.load_img("AHC/" + path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        return image_arr/255.
    
    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path_batch = batches[self.X_col['path']]
        sin_season_batch = batches[self.y_col['sin_date']]
        cos_season_batch = batches[self.y_col['cos_date']]

        X_batch = np.asarray([self.__get_input(x) for x in path_batch])

        y_batch = np.asarray([y for y in zip(sin_season_batch, cos_season_batch)])

        # return X_batch, y0_batch
        return X_batch, y_batch   
        
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)    
        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size

#%% Command-line driver

'''
The DictAction argparse.Action class definition is a mix of codes by 
GitHub users: vadimkantorov and ozcanyarimdunya, with the structure following the latter.
Source: https://gist.github.com/vadimkantorov/37518ff88808af840884355c845049ea 

The strbool argparse type definition is derived from StackOverflow user maxim. 
Source: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
'''

class DictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for each in values:
            try:
                key, value = each.split("=")
                getattr(namespace, self.dest)[key] = value
            except ValueError as ex:
                message = "\nTraceback: {}".format(ex)
                message += "\nError on '{}' || It should be 'key=value'".format(each)
                raise argparse.ArgumentError(self, str(message))

def strbool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in ('true', 't', 'yes', 'y'):
        return True
    elif string.lower() in ('false', 'f', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected. Enter either "True" or "False"')

def main():
    
    parser = argparse.ArgumentParser(
        description='This program generates batches of data for training and testing the ML model.')
    parser.add_argument('df', 
                        type=str,
                        help='Path to df containing list of images with metadata.')
    parser.add_argument('directory', 
                        type=str,
                        help='Path to the root directory of images above the paths listed in df.')
    Required = parser.add_argument_group('Required arguments')
    Optional = parser.add_argument_group('Optional arguments')
    Required.add_argument('--X_col', '-X', '-x',
                        dest='X_col',
                        action=DictAction,
                        metavar="AIfeat=dfcol",
                        default={},
                        required=True,
                        help='Enter key=value pairs to encode model feature to df column variable. ' + \
                            'Will accept -X key1=value1 "key2=value 2". ' + \
                                'Expected keys: path=__')
    Required.add_argument('--y_col', '-y', '-Y',
                        dest='y_col',
                        action=DictAction,
                        metavar="AIfeat=dfcol",
                        default={},
                        required=True,
                        help='Enter key=value pairs to encode model feature to df column variable. ' + \
                            'Will accept -y key1=value1 "key2=value 2". ' + \
                                'Expected keys: id=__, domain=__, sin_date=__, cos_date=__')
    Optional.add_argument('--batch_size', '--batch',
                        type=int, 
                        default=50,
                        help='Integer; number of images in batch for training. Default is 50. ')
    Optional.add_argument('--shuffle', '--s',
                        type=strbool,
                        default=True,
                        help='Str, enter "True" or "t" or "False" or "f". ' +\
                            'Argument can be either upper or lower case. ' +\
                            'Default is True.')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
    
    args = parser.parse_args()
    
    assert os.path.exists(args.df), \
        'df {} does not exist'.format(args.df)
    

if __name__ == '__main__':
    main()

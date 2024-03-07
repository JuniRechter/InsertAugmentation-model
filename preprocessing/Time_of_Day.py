# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:29:07 2024

@author: Juniper Rechter
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import argparse
import time
#import humanfriendly

#%%
os.getcwd() #find current working directory, and set directory below
#os.chdir("C:/Users/hayle/OneDrive/Desktop/test_images/original/AHC")
os.chdir('D:/hayle/Documents/University of Guelph/AI Project Data/crops/AHC_masks')
#%%
night = "AHC/Empty/Night/"
path = "original/"
image = cv2.imread("cropped/bear/GL_RAF002_2022-07-26-15-33-58_07260025_bear_mask.png", cv2.IMREAD_UNCHANGED)
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
img_hsv[:, :, 1].mean()
#%%
def check_ToD(image_directory="image_directory"): #,
       #        CT="CT",
        #       output_directory="output_directory"):
    """
    This function crops and resizes images within a directory and saves them to a specified
    output directory.

    Inputs:
    - image_directory: str, set the directory to pull images from. 
    
    - CT: str, set which camera trap dataset is being used for specific transformations.
    no default set. Enter "AHC" or "MNRF"
        
    - output_directory: str, set the directory to save the resized images to,
      NOTE: Setting the output to the same directory as the input will cause original images 
      to be saved over. Be sure you are using a COPY!

    Output: the cropped and resized images within the output directory.

    """
    data=[]
    for directory, subdirs, files in os.walk(image_directory):
        rel_subdir = os.path.relpath(directory, start=image_directory)
        for f in files:
            if f.endswith('.JPG' or '.jpg' or'.JPEG' or '.jpeg'):
                image = cv2.imread(directory + "/" + f)
                img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                saturation = img_hsv[:, :, 1].mean()
                if saturation <5.0:
                    data.append({'filename':rel_subdir +'/' + f,
                                 'Night':True})
                elif saturation >5.0:
                    data.append({'filename':rel_subdir +'/' + f,
                        'Night':False})
    df = pd.DataFrame(data)
    return df

#%%
def check_crops(image_directory="image_directory"): #,
       #        CT="CT",
        #       output_directory="output_directory"):
    """
    This function crops and resizes images within a directory and saves them to a specified
    output directory.

    Inputs:
    - image_directory: str, set the directory to pull images from. 
    
    - CT: str, set which camera trap dataset is being used for specific transformations.
    no default set. Enter "AHC" or "MNRF"
        
    - output_directory: str, set the directory to save the resized images to,
      NOTE: Setting the output to the same directory as the input will cause original images 
      to be saved over. Be sure you are using a COPY!

    Output: the cropped and resized images within the output directory.

    """
    data=[]
    for directory, subdirs, files in os.walk(image_directory):
        rel_subdir = os.path.relpath(directory, start=image_directory)
        for f in files:
            if f.endswith('.png'):
                image = cv2.imread(directory + "/" + f, cv2.IMREAD_UNCHANGED)
                img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                saturation = img_hsv[:, :, 1].mean()
                if saturation <5.0:
                    data.append({'filename':rel_subdir +'/' + f,
                                 'Night':True})
                elif saturation >5.0:
                    data.append({'filename':rel_subdir +'/' + f,
                        'Night':False})
    df = pd.DataFrame(data)
    return df

#%% Command-line driver

'''
The strbool argparse type definition is derived from StackOverflow user maxim. 
Source: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
'''

def strbool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in ('true', 't', 'yes', 'y'):
        return True
    elif string.lower() in ('false', 'f', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected. Enter either "True" or "False".')

def strformat(string):
    if string.lower() in ('png', '.png', 'p'):
        return "png"
    elif string.lower() in ('jpg', 'jpeg', '.jpg', '.jpeg', 'j'):
        return "jpg"
    else:
        raise argparse.ArgumentTypeError('Image format expected. Enter either "png" or "jpg".')        

def main():
    
    parser = argparse.ArgumentParser(
        description='Program to crop info-bars from CT images and resize in preparation for ML input')
    parser.add_argument('image_directory', 
                        type=str,
                        help='Path to directory of images.')
    parser.add_argument('image_type', 
                        type=str,
                        help='Select if you are checking time-of-day on empty CT images or animal crops.')
    parser.add_argument('output_directory', 
                        type=str,
                        help='Path to the directory to save resized images.')



    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.image_directory), \
        'Image directory {} does not exist'.format(args.image_directory)
    if os.path.exists(args.output_directory):
        print('Warning: output_file {} already exists and will be overwritten'.format(
            args.output_directory))


    start_time = time.time()
    print("Starting check.")
    if args.image_type in ('empty', 'empties'):
        df = check_ToD(image_directory=args.image_directory)
    elif args.image_type in ('crops', 'masks'):
        df = check_crops(image_directory=args.image_directory)

    elapsed = time.time() - start_time

    df.to_csv(args.output_directory + ".csv", index=False)

    print("I'm finished! Finally, I'm a beautiful butterfly!")
    print(('Finished checking Time of Day in {:.3f} seconds.').format(elapsed))

if __name__ == '__main__':
    main()

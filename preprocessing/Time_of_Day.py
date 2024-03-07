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
import humanfriendly

#%%
def check_ToD(image_directory="image_directory"): 

    """
    This function checks images to determine if they were taken at night with infrared or during 
    the day with full colour.
    Note that this version works for full three-channel CT images, but not for image masks/crops. 
    
    Inputs:
    - image_directory: str, set the directory to pull images from. 
    
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
def check_crops(image_directory="image_directory"): 
    """
    This function checks animal mask crops to determine if they were taken at night with infrared or during 
    the day with full colour.
    Note that this version works for four-channeled .PNG animal masks, but not for the original CT images. 
    
    Inputs:
    - image_directory: str, set the directory to pull images from. 
    
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

def main():
    
    parser = argparse.ArgumentParser(
        description='Program to crop info-bars from CT images and resize in preparation for ML input')
    parser.add_argument('image_directory', 
                        type=str,
                        help='Path to directory of images.')
    parser.add_argument('image_type', 
                        type=str,
                        help='Str, enter if you are checking original CT images or animal crops.' +\
                       'Enter either "empties" or "masks".)
    parser.add_argument('save', 
                        type=str,
                        help='Enter filename for the created CSV file.')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.image_directory), \
        'Image directory {} does not exist'.format(args.image_directory)
    if os.path.exists(args.output_directory):
        print('Warning: output_file {} already exists and will be overwritten'.format(
            args.output_directory))
    assert args.image_type in ('empty', 'empties', 'CT', 'crops', 'masks'), \
        'Cannot select function for "{}". Please enter either "CT" or "empties" for full CT image, or "masks" for animal masks.'.format(args.image_type)


    start_time = time.time()
    print("Starting check.")
    if args.image_type in ('empty', 'empties', 'CT'):
        df = check_ToD(image_directory=args.image_directory)
    elif args.image_type in ('crops', 'masks'):
        df = check_crops(image_directory=args.image_directory)

    elapsed = time.time() - start_time

    df.to_csv(args.save + ".csv", index=False)

    print("I'm finished! Finally, I'm a beautiful butterfly!")
    print(('Finished checking Time of Day in {:.3f} seconds.').format(elapsed))

if __name__ == '__main__':
    main()

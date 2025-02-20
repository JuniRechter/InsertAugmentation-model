# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:29:07 2024
@author: Juniper Rechter

This file contains functions checking if either an original camera trap (CT) image (.JPG)
or a SAM-animal mask crop (.PNG) was taken with infrared at nighttime. These functions will
open images and check the mean saturation of the image; if saturation < 5, the image was likely 
taken at night with infrared, and the script will assign Night: True.
This will output a .CSV file (containing the image filename and its boolean Night value) that 
can be merged with other dataframes along the filename to update the main dataset. 

This script is quite fast: for ~50,000 full-sized crops (~20GB images), it only requires ~1GB memory, and will
complete Time-of-Day check in ~10-15 mins. Can easily be run on laptops and potatoes. 

Command-line arguments:
    - image_directory: str, path to directory of images.
    - image_type: str, for original CT image, enter one of: "CT", "empties", "empty"
                       for SAM-animal mask crop, enter one of: "masks", "crops"
    - save: str, enter filename for the created CSV file.

Example:
    python Time_of_Day.py AHC AHC_masks_ToD --crops
"""

import os
import sys
import pandas as pd
import cv2
import argparse
import time
import humanfriendly
#%%
CT_EXT = ['jpg', '.jpeg']
MASK_EXT = ['.png']
#%%
def check_ToD(image_directory="image_directory", crops=False): 

    """
    This function checks images to determine if they were taken at night with infrared or during 
    the day with full colour.
    Note that this version works for full three-channel CT images, but not for image masks/crops. 
    
    Inputs:
    - image_directory: str, set the directory to pull images from. 
    - crops: bool, if True, function will check animal cutout (crops/masks) PNG files instead, and load four channels (RGBA).
    
    """
    data=[]
    
    if crops==False:
        for directory, subdirs, files in os.walk(image_directory):
            rel_subdir = os.path.relpath(directory, start=image_directory)
            for f in files:
                if f.lower().endswith(tuple(CT_EXT)):
                    image = cv2.imread(directory + "/" + f)
                    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    saturation = img_hsv[:, :, 1].mean()
                    if saturation <5.0:
                        data.append({'filename':rel_subdir +'/' + f,
                                     'Night':True})
                    elif saturation >5.0:
                        data.append({'filename':rel_subdir +'/' + f,
                            'Night':False})
    else:
        for directory, subdirs, files in os.walk(image_directory):
            rel_subdir = os.path.relpath(directory, start=image_directory)
            for f in files:
                if f.lower().endswith(tuple(MASK_EXT)):
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
    parser.add_argument('save', 
                        type=str,
                        help='Enter filename for the created CSV file.')
    parser.add_argument('--crops', 
                        action='store_true',
                        help='Include argument if checking ToD of cropped animals.')
    
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
    df = check_ToD(image_directory=args.image_directory, args.crops)

    elapsed = time.time() - start_time

    df.to_csv(args.save + ".csv", index=False)

    print("I'm finished! Finally, I'm a beautiful butterfly!")
    print(('Finished checking Time of Day in {}.').format(humanfriendly.format_timespan(elapsed)))

if __name__ == '__main__':
    main()

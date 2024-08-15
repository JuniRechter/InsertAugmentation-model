# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:28:43 2024

@author: Juniper Rechter
"""
import os
import sys
import cv2
import pandas as pd
from imgaug import augmenters as iaa
import argparse
import time
import humanfriendly

#%% Set imgaug function to resize animal cutouts to 20% of original size.
shrink = iaa.Resize(0.2, interpolation="area")

#%%
def shrink_crops(df_file):
    df = pd.read_csv(df_file)
    og_size=[]
    new_size = []
    area_list=[]
    for i in df.index:
        cpath=df['filename'][i]
        crop = cv2.imread(('crops/' + cpath), cv2.IMREAD_UNCHANGED)
        H = crop.shape[0] #set the image height
        W = crop.shape[1] #set the image width
        og_size.append(tuple((W,H)))
        crop_aug = shrink.augment_image(crop)
        H = crop_aug.shape[0] #set the image height
        W = crop_aug.shape[1] #set the image width
        area = H*W
        new_size.append(tuple((W,H)))
        area_list.append(area)
        cv2.imwrite("crops/" + cpath, cv2.cvtColor(crop_aug, cv2.COLOR_BGR2RGBA))
        if (i+1)%50==0:
            print(((df['filename'][i]) + " segmented and saved. \n Image {} of {}.").format((i+1), len(df)), flush=True)
    df['new_size']=new_size
    df['area']=area_list
    df.to_csv('SHRUNK_' + df_file, index=False)
#%%
def main():

    parser = argparse.ArgumentParser(
        description='Program to shrink segmented animal masks.')
    parser.add_argument('df', 
                        type=str,
                        help='Path to directory of images.')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.df), \
        'df {} does not exist'.format(args.df)            

    start_time = time.time()

    shrink_crops(df_file = args.df)

    elapsed = time.time() - start_time
    images_per_second=len(df)/elapsed
    print("I'm finished! Finally, I'm a beautiful butterfly!")
    print(('Finished shrinking {} images in {} ({:.2f} images per second).').format(len(df), humanfriendly.format_timespan(elapsed), images_per_second))

if __name__ == '__main__':
    main()

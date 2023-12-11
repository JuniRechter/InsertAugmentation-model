# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:04:42 2023

@author: Juniper Rechter

"""
#%%
from PIL import Image, ImageDraw, ImageOps
import os

import argparse
import sys
import time
#import humanfriendly
#%%
def cropresize(image_directory="image_directory",
               CT="CT",
               blackout=True,
               upper=0,
               lower=0,
               resize=False,
               new_size=(224,224),
               save_format="png",
               output_directory="output_directory"):
    """
    This function crops and resizes images within a directory and saves them to a specified
    output directory.

    Inputs:
    - image_directory: str, set the directory to pull images from. 
    
    - CT: str, set which camera trap dataset is being used for specific transformations.
    no default set. Enter "AHC" or "MNRF"
    
    - blackout: bool, set whether infobars should be blacked out.
    default = True, will add black boxes over the metadata text on the CT images.
    
    - upper: int, how many pixels to sheer off the upper margin, 
    default = 0, will not crop from the top of the image.
    
    - lower: int, how many pixels to sheer off the lower margin, 
    default = 0, will not crop from the top of the image.
    
    - resize: bool, set whether or not image needs to be resized.
    default = False. If True, new_size defaults to (800, 450)
    
    - new_size = tuple, set the dimensions for resizing the image,
    default (W, H) = (800, 450) pixels.
      
    - save_format: str, save image as either a "png" or a "jpg".
    default = "png". NOTE: PNG's tend to save in higher quality, 
    whilst JPGs often pixelate between sharp borders.
    
    - output_directory: str, set the directory to save the resized images to,
      NOTE: Setting the output to the same directory as the input will cause original images 
      to be saved over. Be sure you are using a COPY!

    Output: the cropped and resized images within the output directory.

    """

    for directory, subdirs, files in os.walk(image_directory):
        rel_subdir = os.path.relpath(directory, start=image_directory)
        for f in files:
            if f.endswith('.JPG' or '.jpg' or'.JPEG' or '.jpeg'):
                img = Image.open(directory + "/" + f)
                draw= ImageDraw.Draw(img)
                if CT=="MNRF":
                    draw.ellipse((1823, 1472, 2033, 1536), fill = (0,0,0), outline=None) #Black Reconyx logo
                    if blackout==True:
                        draw.rectangle((0, 0, 2048, 32), fill = (0,0,0), outline=None) #Black top infobar
                        draw.rectangle((0, 1504, 2048, 1536), fill = (0,0,0), outline=None) #Black bottom infobar
                    else:
                        pass
                    img=ImageOps.pad(img, size=(2048,1626), color="black") #Vertical padding
                    img=ImageOps.pad(img, size=(2752,1626), color="black") #Horizontal padding 
                    img=ImageOps.pad(img, size=(2752,2752), color="black") #Squaring
                elif CT=="AHC":
                    if blackout==True:
                        draw.rectangle((0, 1231, 2304, 1296), fill = (0,0,0), outline=None) #Black infobar
                    else:
                        pass
                    img=ImageOps.pad(img, size=(2304,1361), color="black", centering=(0.5,1)) #Add top infobar
                    img=ImageOps.pad(img, size=(2304,2304), color="black") #Squaring
                    draw= ImageDraw.Draw(img)
                    draw.ellipse((1821, 1741, 1996, 1795), fill = (0,0,0), outline=None) #Fake black logo
                img = img.crop((0, int(0+upper), img.size[0], int(img.size[1]-lower)))
                if resize==True:
                    img = img.resize(new_size)
                else:
                    pass
                if not os.path.exists(output_directory + "/" + rel_subdir):
                    os.mkdir(output_directory + "/" + rel_subdir)
                if save_format=="png":
                    img.save(output_directory + "/" + rel_subdir + "/" + f + ".png")
                elif save_format=="jpg":
                    img.save(output_directory + "/" + rel_subdir + "/" + f)

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
    parser.add_argument('output_directory', 
                        type=str,
                        help='Path to the directory to save resized images.')
    parser.add_argument('CT', 
                        type=str,
                        help='Str, enter which camera trap set is being used for specific transformations. ' + \
                            'Enter either AHC or MNRF')
    parser.add_argument('blackout',
                        type=strbool,
                        default=True,
                        help='Str, enter "True" or "t" or "False" or "f" to blackout info bars. ' +\
                            'Argument can be either upper or lower case. ' +\
                            'Default is False.')
    parser.add_argument('save_format',
                        type=strformat,
                        help='Str, enter preferred image format, either png or jpg. ' + \
                            'NOTE: PNGs tend to save in higher quality but will result in a copy; ' +\
                            'JPGs often pixelate over sharp borders but will replace original image.')
    parser.add_argument('resize',
                        type=strbool,
                        default=False,
                        help='Str, enter "True" or "t" or "False" or "f". ' +\
                            'Argument can be either upper or lower case. ' +\
                            'Default is False.')
    parser.add_argument('--new_size',
                        type=int,
                        metavar=('WIDTH','HEIGHT'),
                        nargs=2,
                        default=[224,224], 
                        help='Integers, should contain Width and Height, separated by a " ".')
    parser.add_argument('--upper', '--u',
                        type=int, 
                        default=0,
                        help='Integer; number of pixels to crop from the top of the image.')
    parser.add_argument('--lower', '--l',
                        type=int, 
                        default=0,
                        help='Integer; number of pixels to crop from the bottom of the image.')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.image_directory), \
        'Image directory {} does not exist'.format(args.image_directory)       
    if len(args.new_size) != 2:
        print('New_size should contain width and height, separated by a " ".')
    elif any(dim <1 for dim in args.new_size):
        print("Both width and height need to have positive values > 0.")
    assert os.path.exists(args.output_directory), \
        'Output directory {} does not exist'.format(args.output_directory)
    if os.path.exists(args.output_directory):
        print('Warning: output_file {} already exists and will be overwritten'.format(
            args.output_directory))
        
    start_time = time.time()

    cropresize(image_directory=args.image_directory,
               CT=args.CT,
               blackout=args.blackout,
               upper=args.upper,
               lower=args.lower,
               resize=args.resize,
               new_size=tuple(args.new_size),
               save_format=args.save_format,
               output_directory=args.output_directory)

    elapsed = time.time() - start_time

    print("I'm finished! Finally, I'm a beautiful butterfly!")
    print(('Finished resizing images in {:.3f} seconds.').format(elapsed))
#    print(('Finished resizing images in {}.').format(humanfriendly.format_timespan(elapsed)))

if __name__ == '__main__':
    main()
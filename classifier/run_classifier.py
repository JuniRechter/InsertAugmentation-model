# -*- coding: utf-8 -*-
"""
Created on Mon Jun 3 10:38:14 2024

@author: Juniper Rechter

This file contains functions for running the trained AHC species classifier model, 
with options to request more or less data in the output. 
Currently, this program uses CSV files (either provided or generated from a directory)
to set the list of images to be tested. 

Command-line arguments:
    Required
    - classifier_file: str, path to classifier model file (ending in ".keras"). 
                       
    - image_paths: str, str, path to either a CSV file containing list of paths to images, or a directory
                       containing images to be evaluated.
                       
    - output_file: str, path to output CSV results file, should end with a .csv extension.

    Optional
    --recursive: include argument to search recursively through the provided directory and all subdirectories.
    --column: str, name of the column/variable containing file paths. Can use if providing a CSV.
                       Pathing within CSV must be relative to directory the program is run from.
                       Default is "full_path".

    --max_conf: include argument to request the confidence score of the highest scoring prediction within
                       the output CSV file.

    --top5: bool, would you like the model to output the top 5 predictions per image? Enter "True" or "t" or 
                       "False" or "f"; Argument can be either upper or lower case. 
                       Default is True.

    --threshold: float, confidence threshold between 0.0 and 1.0; only provide predictions above this threshold, 
                       else prediction will return as "unknown" to flag user for verification. 
                       Default is None (disabled).
                       
    --full_results: include argument to request the full set of predictions, providing an ordered list of 
                       15 classes (14 species and 1 "unknown" class) and their respective confidence scores,
                       within the output CSV file.


Example:
    python Time_of_Day.py AHC masks AHC_masks_ToD
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from PIL import ImageOps
import keras
from tensorflow.keras.optimizers import Adam
import training.ImageDataGenerators as idg
import argparse
import time
import humanfriendly

#%% #Set constant variables for function use.

image_extensions = ['.jpg', '.jpeg', '.gif', '.png']

DEFAULT_15SPP_CLASSES = ['deer', 'fox', 'moose', 'bear', 'sandhill crane',
                         'turkey', 'raccoon', 'crow_raven', 'wolf', 'hare',
                         'coyote', 'squirrel grey', 'domestic dog', 'bobcat', 
                         'unknown']

#%%
def generate_file_df(image_directory, recursive=False):
    '''
    If no CSV file is provided with a list of image paths, this function will search a provided directory 
    and create a CSV file containing the list for model prediction.

    Inputs:
        - image_directory: str, set the directory to pull images from. 
    
    '''
    data=[]
    if recursive == True:
        for directory, subdirs, files in os.walk(image_directory):
            rel_subdir = os.path.relpath(directory, start=image_directory)
            for f in files:
                if f.lower().endswith(tuple(image_extensions)):
                    data.append({'RootFolder': image_directory,
                                 'File': f, 'RelativePath': rel_subdir,
                                 'path': os.path.join(rel_subdir, f),
                                 'full_path': os.path.join(image_directory, rel_subdir, f)})
    else:
        for directory, subdirs, files in os.walk(image_directory):
            for f in files:
                if f.lower().endswith(tuple(image_extensions)):
                    data.append({'RootFolder': image_directory,
                                 'File': f,
                                 'full_path': os.path.join(image_directory, f)})
            break
    df = pd.DataFrame(data)
    return df

#%%
def generate_results_df(df, y_pred, output_file, max_conf=False, top_5=False, threshold=None, full_results=False):
    '''
    This function takes a provided DataFrame (either from user-given CSV file or a generated one) and adds 
    model predictions and corresponding classes. It will provide the top prediction in one column, and the 
    top-5 predictions and their corresponding confidence scores (%) in another column. The full list of 
    predictions (14 species and 1 unknown class) and confidence scores can be given if requested. 
  
    Inputs:
    - df: takes the CSV file generated or provided that contains a list of all images to be tested.
    - y_pred: array, the full array of results output by model prediction.
    - output_file: str, path to output and save CSV results file, should end with a .csv extension.
    - max_conf: bool, would you like the model to output the confidence score for the highest scoring prediction?
    - top_5: bool, would you like the model to output the top five predictions (and confidence scores) per image?
    - threshold: float, confidence threshold between 0.0 and 1.0; only provide predictions above this threshold, 
                 else an image prediction will return as "unknown" to flag user for verification. 
                 Default is None (disabled).
    - full_results: bool, would you like the model to output confidence scores for all 15 classes?
                    Default is False.
  
    '''
   
    class_dict = dict(enumerate(DEFAULT_15SPP_CLASSES))
    
    class_pred = np.array([])
    class_pred = np.append(class_pred, np.argmax(y_pred, axis=1))

    pred_class=(class_pred.astype(int)).tolist()
    df['ClassPred']=pred_class
    df['SpeciesPred'] = df['ClassPred'].apply(lambda x: class_dict[x])
    full_prediction = []
    top5_preds = []
    top1_conf = []
    for i in df.index:
        species_pred=df['SpeciesPred'][i]
        image_preds = (y_pred[i])
        sorted_preds = sorted(image_preds, reverse=True)
        sorted_percents = sorted(zip([f'{pred*100:.2f}%' for pred in image_preds], DEFAULT_15SPP_CLASSES), reverse=True)
        top_conf = (f'{max(image_preds)*100:.2f}%')
        top5=sorted_percents[:5]
        if threshold is not None:
            if sorted_preds[0] < threshold:
                df['SpeciesPred'][i]="unknown - {}".format(species_pred)
        top1_conf.append(top_conf)
        top5_preds.append(top5)
        full_prediction.append(sorted_percents)
    
    if max_conf==True:
        df['Pred_Confidence']=top1_conf
    if top_5==True:
        df['Top5_Predictions'] = top5_preds
    if full_results==True:
        df['Full_Prediction_Scores']=full_prediction

    df.drop(columns=['ClassPred', 'dummy_id'], inplace=True)
    
    df.to_csv(output_file, index=False)
    print('Results saved to {}'.format(output_file))

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
        raise argparse.ArgumentTypeError('Boolean value expected. Enter either "True" or "False"')

def main():
    
    parser = argparse.ArgumentParser(
        description='Module to run AHC classifier on batch of images from a directory or CSV containing filenames.')
    parser.add_argument('classifier_file',
                        type=str,
                        help='Path to classifier model file (ending in ".keras").')
    parser.add_argument('image_paths', 
                        type=str,
                        help='str, path to either a CSV file containing list of paths to images, ' +\
                             'or a directory containing images to be evaluated.')
    parser.add_argument('output_file',
                        type=str,
                        help="Path to output CSV results file, should end with a .csv extension.")
    Optional = parser.add_argument_group('Optional arguments')
    Optional.add_argument('--recursive',
                          action='store_true',
                          help='Include argument to search recursively through the provided directory and all subdirectories.')
    Optional.add_argument('--column',
                          type=str,
                          default=None,
                          help='str, name of the file path column/variable if providing a CSV. ' +\
                               'Image pathing within CSV must be relative to directory program is run from.' +\
                               'Default is "full_path')
    Optional.add_argument('--include_max_conf',
                          action='store_true',
                          help='Include argument to request the confidence score of the highest scoring prediction.')
    Optional.add_argument('--top5',
                          type=strbool,
                          action='store_true',
                          help='Include argument to request the top 5 predictions (and their corresponding confidence scores) per image.')
    Optional.add_argument('--threshold',
                          type=float,
                          default=None,
                          help='float, Confidence threshold between 0.0 and 1.0; only provide predictions above this ' +\
                               'threshold, else prediction will return as "unknown" to flag user for verification. ' +\
                               'Default is None (disabled).')
    Optional.add_argument('--full_results',
                          action='store_true',
                          help='include argument to request the full set of predictions, providing an ordered list of ' +\
                               '15 classes (14 species and 1 "unknown" class) and their respective confidence scores, ' +\
                               'within the output CSV file.')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
    
    args = parser.parse_args()
    
    assert os.path.exists(args.classifier_file), \
        'classifier {} does not exist'.format(args.classifier_file)
    if os.path.exists(args.output_file):
        print('Warning: output_file {} already exists and will be overwritten.'.format(args.output_file) +\
              '\nIf you provided a CSV file, this will only add columns to original file.')
    assert 0.0 < args.threshold <= 1.0, \
        'Confidence threshold needs to be between 0.0 and 1.0.'
    
    if os.path.isdir(args.image_paths):
        test_df = generate_file_df(args.image_paths, recursive=args.recursive)
        if len(test_df) >0:
            print('{} images found in the input directory.'.format(len(test_df)))
            column = "full_path"
        else:
            print('No images found in the directory {}. Exiting.'.format(args.image_paths))
    elif os.path.isfile(args.image_paths):
        assert os.path.exists(args.image_paths), \
            'df {} does not exist'.format(args.image_paths)
        test_df = pd.read_csv(args.image_paths)
        print('Loaded {} image filenames from CSV file {}.'.format(len(test_df), args.image_paths))
        if args.column is not None:
            assert args.column in test_df.columns, \
                'Column {} does not exist within CSV file {}.'.format(args.column, args.image_paths)
            column = args.column
        else:
            assert "full_path" in test_df.columns, \
                'Column "full_path" does not exist within CSV file {}. '.format(args.image_paths) +\
                    '\nPlease specify which column contains the list of file paths.'
            column = "full_path"
    else:
        raise ValueError('image_file specified is not a CSV dataframe, or a directory.')
    
    test_df['dummy_id']=0
    testgen = idg.PredictionDataGenerator(test_df,
                                          batch_size=1,
                                          y_col = {'id': 'dummy_id'},
                                          X_col = {'path': column},
                                          shuffle = False)

    if args.classifier_file == "AHC_SPP15_ID_mdl_wts.keras":
        model_name = "AHC_SPP15"
        
    print(("Preparing to load {} classifer model.").format(model_name))
    load_time = time.time()
    AHCmodel = tf.keras.models.load_model(args.classifier_file, compile=False)
    #Model compilation required due to some incompatibility between tf and keras.
    AHCmodel.compile(optimizer=Adam(learning_rate=1e-4), 
                     loss="categorical_crossentropy", 
                     metrics=["accuracy"])
    elapsed = time.time() - load_time
    print(('{} classifier model loaded in {}.').format(model_name, humanfriendly.format_timespan(elapsed)))
    
    start_time = time.time()
    
    results = AHCmodel.predict(testgen, verbose=2)
    
    model_run_time = time.time() - start_time
    n_images = results.shape[0]
    images_per_second = n_images / model_run_time
    print('Finished predictions for {} images in {} ({:.2f} images per second)'.format(n_images, 
                                                                                       humanfriendly.format_timespan(model_run_time), 
                                                                                       images_per_second))
    
    generate_results_df(test_df, 
                        results, 
                        output_file=args.output_file,
                        max_conf=args.max_conf,
                        top_5=args.top5,
                        threshold=args.threshold, 
                        full_results=args.full_results)
    
    print("Finished! \nThank you and have a nice day!")
    
    if __name__ == '__main__':
        main()

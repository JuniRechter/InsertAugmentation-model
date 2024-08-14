# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:38:14 2024

@author: Juniper Rechter

Command-line arguments:
    - classifier_file: str, path to classifier model file (ending in ".keras"). 
                       Required argument.
    - df: str, path to CSV file containing list of images to be evaluated.
                       Argument mutually exclusive with directory; must provide either CSV or directory.
    - directory: str, path to directory of images.
                       Argument mutually exclusive with df, Must provide either directory or CSV.
    - output_file: str, path to output CSV results file, should end with a .csv extension.
                       Required argument.
    --paths: str, name of the column/variable containing file paths. Can use if providing a CSV.
                       Pathing within CSV must be relative to directory the program is run from.
                       Optional argument.
    --top5: bool, would you like the model to output the top 5 predictions per image? Enter "True" or "t" or 
                      "False" or "f"; Argument can be either upper or lower case. Default is True.
                       Optional argument.
    --threshold: float, confidence threshold between 0.0 and 1.0; only provide predictions above this threshold, 
                       else prediction will return as "unknown" to flag user for verification. Default is None (disabled).
                       Optional argument.

Examples:
    python run_classifier.py path/to/AHC_SPP15_ID_mdl_wts.keras path/to/images results_file.csv --threshold 0.7
    python run_classifier.py path/to/AHC_SPP15_ID_mdl_wts.keras path/to/dataframe.csv results_file.csv --paths filenames
"""

import os
import sys
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from PIL import ImageOps
from tensorflow.keras.optimizers import Adam
import training.ImageDataGenerators as idg
import argparse
import time
import humanfriendly

#%%
#%%
def generate_file_df(image_directory):
    '''
    If no CSV file is provided with a list of image paths, this function will search a provided directory 
    and create a CSV file containing the list for model prediction.
  
    Inputs:
        - image_directory: str, set the directory to pull images from. 
    
    '''
    data=[]
    for directory, subdirs, files in os.walk(image_directory):
        rel_subdir = os.path.relpath(directory, start=image_directory)
        for f in files:
            if f.endswith('.JPG' or '.jpg' or'.JPEG' or '.jpeg'):
                data.append({'RootFolder': image_directory,
                             'File': f, 'RelativePath': rel_subdir,
                             'path':rel_subdir +'/' + f,
                             'full_path': image_directory + '/' + rel_subdir + '/' + f})
    df = pd.DataFrame(data)
    return df

#%%
def generate_results_df(df, y_pred, threshold=None, full_results=False):
    '''
    This function takes a provided DataFrame (either from user-given CSV file or a generated one) and adds 
    model predictions and corresponding classes. It will provide the top prediction in one column, and the 
    top-5 predictions and their corresponding confidence scores (%) in another column. The full list of 
    predictions (14 species and 1 unknown class) and confidence scores can be given if requested. 
  
    Inputs:
    - df:
    - y_pred: array, the full array of results output by model prediction
    - threshold: float, confidence threshold between 0.0 and 1.0; only provide predictions above this threshold, 
                 else an image prediction will return as "unknown" to flag user for verification. 
                 Default is None (disabled).
    - full_results: bool, would you like the model to output confidence scores for all 15 classes?
                    Default is False.
  
    '''
  
    classes = ['deer', 'fox', 'moose', 'bear', 'sandhill crane',
               'turkey', 'raccoon', 'crow_raven', 'wolf', 'hare',
               'coyote', 'squirrel grey', 'domestic dog', 'bobcat', 
               'unknown']
    class_dict = dict(enumerate(classes))
    
    class_pred = np.array([])
    class_pred = np.append(class_pred, np.argmax(y_pred, axis=1))

    pred_class=(class_pred.astype(int)).tolist()
    df['ClassPred']=pred_class
    df['SpeciesPred'] = df['ClassPred'].apply(lambda x: class_dict[x])
    full_prediction = []
    percentages = []
    top5_preds = []
    length = y_pred.shape[0]
    for i in range(0, length):
        image_preds = (y_pred[i])
        percentage = [f'{pred*100:.3f}%' for pred in image_preds]
        top5=sorted(zip(([f'{pred*100:.2f}%' for pred in image_preds]), classes), reverse=True)[:5]
        if threshold is not None:
            if top5[0] < threshold:
                df['SpeciesPred'][i]="unknown"
        top5_preds.append(top5)
        full_prediction.append(image_preds)
        percentages.append(percentage)

    df['Top5_Preds'] = top5_preds
    if full_results==True:
        df['full_prediction_score']=full_prediction
        df['Percentages']=percentages
    else:
        df.drop(columns='ClassPred', inplace=True)
      
    df.drop(columns='dummy_id', inplace=True)
    return df

#%%
AHCmodel = tf.keras.models.load_model('D:/hayle/Documents/University of Guelph/AI Project Data/Models/AHC/Results/FINAL AHC Model/15SPP_Expo0a_FINALMODEL/AHC_SPP15_ID_mdl_wts.keras', compile=False)
AHCmodel.compile(optimizer=Adam(learning_rate=1e-4), 
                 loss="categorical_crossentropy", 
                 metrics=["accuracy"])
#%%
test_df = generate_file_df("C:/Users/hayle/OneDrive/Desktop/test_images/original/testing")
test_df['dummy_id']=0
testgen = PredictionDataGenerator(test_df,
                                  batch_size=1,
                                  y_col = {'id': 'dummy_id'},
                                  X_col = {'path': 'full_path'},
                                  shuffle = False)
y_pred = AHCmodel.predict(testgen, verbose=2)

#%%
image = tf.keras.utils.load_img("original/AHC/Spring/SC_TRV003_2021-05-31-06-20-48_DCIM0761 (2).JPG")
w, h = image.size
image=ImageOps.pad(image, size=(w,h+64), color="black", centering=(0.5,1)) #Add top infobar 
image.save("centering050.JPG")
results_df = generate_results_df(test_df, y_pred)
results_df.to_csv('C:/Users/hayle/OneDrive/Desktop/test_images/original/Results_df.csv', index=False)
results_df[['path', 'SpeciesPred', 'Percentages']]
#%%
class_dict = {0:'deer', 1:'fox', 2:'moose', 3:'bear', 4:'sandhill crane',
              5:'turkey', 6:'raccoon', 7:'crow_raven', 8:'wolf', 9:'hare',
              10:'coyote', 11:'squirrel grey', 12:'domestic dog', 13:'bobcat', 
              14:'unknown'}
classes = ['deer', 'fox', 'moose', 'bear', 'sandhill crane',
              'turkey', 'raccoon', 'crow_raven', 'wolf', 'hare',
              'coyote', 'squirrel grey', 'domestic dog', 'bobcat', 
              'unknown']
class_dict = dict(enumerate(classes))
print(class_dict)
class_pred = np.array([])
class_pred = np.append(class_pred, np.argmax(y_pred, axis=1))
results_df = pd.read_csv('C:/Users/hayle/OneDrive/Desktop/test_images/original/AHC/Fall/Results_df.csv')
perc = results_df['Percentages']
img1 = y_pred[0]
results_df['path'][0]
[f'{x*100:.3f}%' for x in img1]
testing = zip(classes, img1)
testing[0]
sorted(testing, reverse=True)[:3]
top5=sorted(zip(classes, img1), reverse=True)[:5]
top5
for i in results_df.index:
    perc = results_df['Percentages'][i]
    zip(classes, perc)
    perc = results_df['SpeciesPred'][i]
sorted(zip())
#%%
output_file = "dummy_file.csv"
print('Warning: output_file {} already exists and will be overwritten. '.format(output_file) +\
      'If same as provided CSV file, original columns will remain intact.')

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
    MutualExclusiveArgs = parser.add_mutually_exclusive_group(required=True)
    MutualExclusiveArgs.add_argument('df', 
                                     type=str,
                                     help='str, Path to CSV file containing list of images to be evaluated. ' +\
                                          'Must provide either CSV or directory.')
    MutualExclusiveArgs.add_argument('directory',
                                     type=str,
                                     help='str, Path to the directory of images to be evaluated. ' +\
                                          'Must provide either directory or CSV.')
    parser.add_argument('output_file',
                        type=str,
                        help="Path to output CSV results file, should end with a .csv extension.")
    Optional = parser.add_argument_group('Optional arguments')
    Optional.add_argument('paths',
                          type=str,
                          default=None,
                          help='str, name of the file path column/variable if providing a CSV. ' +\
                               'Image pathing within CSV must be relative to directory program is run from.' +\
                               'Default is "full_path')
    Optional.add_argument('--top5',
                          type=strbool,
                          default=True,
                          help='str, Would you like the model to output the top 5 predictions per image? ' +\
                               'Enter "True" or "t" or "False" or "f"; Argument can be either upper or lower case. ' +\
                               'Default is True.')
    Optional.add_argument('--threshold',
                          type=float,
                          default=None,
                          help='float, Confidence threshold between 0.0 and 1.0; only provide predictions above this ' +\
                               'threshold, else prediction will return as "unknown" to flag user for verification. ' +\
                               'Default is None (disabled).')
    Optional.add_argument('--full_results',
                          action='store_true',
                          help='float, Confidence threshold between 0.0 and 1.0; only provide predictions above this ' +\
                               'threshold, else prediction will return as "unknown" to flag user for verification. ' +\
                               'Default is None (disabled).')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
    
    args = parser.parse_args()
    
    assert os.path.exists(args.classifier_file), \
        'classifier {} does not exist'.format(args.classifier_file)
    if os.path.exists(args.output_file):
        print('Warning: output_file {} already exists and will be overwritten.'.format(args.output_file) +\
              'If you provided a CSV file, this will only add columns to original file.')
    
    if args.df is None and args.directory is not None:
        assert os.path.exists(args.directory), \
            'directory {} does not exist'.format(args.directory)
        test_df = generate_file_df("C:/Users/hayle/OneDrive/Desktop/test_images/original/testing")
    elif args.df is not None and args.directory is None:
        assert os.path.exists(args.df), \
            'df {} does not exist'.format(args.df)
        test_df = pd.read_csv(args.df)
        if args.paths is not None:
            assert args.df in test_df.columns, \
                'df column {} does not exist.'.format(args.paths)
            path = args.paths
        else:
            path = "full_path"
    else:
        assert args.df is not None and args.directory is not None, \
            'Must provide either a CSV dataframe or a directory.'
    
    test_df['dummy_id']=0
    n_images = len(test_df)
    testgen = idg.PredictionDataGenerator(test_df,
                                          batch_size=1,
                                          y_col = {'id': 'dummy_id'},
                                          X_col = {'path': path},
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
    print(('{} model loaded in {}.').format(model_name, humanfriendly.format_timespan(elapsed)))
    
    start_time = time.time()
    
    results = AHCmodel.predict(testgen, verbose=2)
    
    model_run_time = time.time() - start_time
    
    images_per_second = n_images / model_run_time
    print('Finished predictions for {} images in {} ({:.2f} images per second)'.format(n_images, 
                                                                                       humanfriendly.format_timespan(model_run_time), 
                                                                                       images_per_second))
    
    results_df = generate_results_df(test_df, results, args.threshold, full_results=args.full_results)
    results_df.to_csv(args.output_file, index=False)
    
if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:18:20 2023

@author: Juniper Rechter
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import keras.backend as K
from tensorflow.keras.applications import densenet
import tensorflow.keras.layers as kl
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Layer
import keras_tuner

import itertools
import sklearn.metrics

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from keras_tuner.tuners import BayesianOptimization
from keras_tuner import Objective
from keras_tuner_cv.utils import pd_inner_cv_get_result

#Optimizer options
from tensorflow.keras.optimizers import Adam

import argparse
from datetime import datetime
import time
import humanfriendly
#%% Load dataset
#Gradient reversal layer for DANN (Ganin et al. 201x)
@tf.custom_gradient
def grad_reverse(x):
    result = x
    def custom_grad(dy):
        return -dy
    return result, custom_grad

class ReversalLayer(Layer):
    def __init__(self, **kwargs):
        super(ReversalLayer, self).__init__()
    def call(self, x):
        return grad_reverse(x)
#%%
# https://github.com/gabriben/metrics-as-losses/blob/main/VLAP/weightedF1.py
def macroF1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

def cat_wf1(true_y, pred_y):
    """Compute the weighted F1-score on a batch of observations (average F1 across labels)
    
    Args:
        true_y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        pred_y (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
#Daniel Moller - StackOverflow (o with umlaut above)
    #for metrics include these two lines, for loss, don't include them
    #these are meant to round 'pred' to exactly zeros and ones
    predLabels = K.argmax(pred_y, axis=-1)
    pred_y = K.one_hot(predLabels, 6) 


    ground_positives = K.sum(true_y, axis=0) + K.epsilon()       # = TP + FN
    pred_positives = K.sum(pred_y, axis=0) + K.epsilon()         # = TP + FP
    true_positives = K.sum(true_y * pred_y, axis=0) + K.epsilon()  # = TP
        #all with shape (4,)
    
    precision = true_positives / pred_positives 
    recall = true_positives / ground_positives
        #both = 1 if ground_positives == 0 or pred_positives == 0
        #shape (4,)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        #still with shape (4,)

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives) 
    weighted_f1 = K.sum(weighted_f1)

    
    return weighted_f1 #for metrics, return only 'weighted_f1'

def weighted_F1(true_y, pred_y):
    """Compute the weighted F1-score on a batch of observations (average F1 across labels)
    
    Args:
        true_y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        pred_y (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
#Daniel Moller - StackOverflow (o with umlaut above)
    #for metrics include these two lines, for loss, don't include them
    #these are meant to round 'pred' to exactly zeros and ones
    predLabels = K.argmax(pred_y, axis=-1)
    pred_y = K.one_hot(predLabels, 6) 


    ground_positives = K.sum(true_y, axis=0) + K.epsilon()       # = TP + FN
    pred_positives = K.sum(pred_y, axis=0) + K.epsilon()         # = TP + FP
    true_positives = K.sum(true_y * pred_y, axis=0) + K.epsilon()  # = TP
        #all with shape (4,)
    
    precision = true_positives / pred_positives 
    recall = true_positives / ground_positives
        #both = 1 if ground_positives == 0 or pred_positives == 0
        #shape (4,)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        #still with shape (4,)

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives) 
    weighted_f1 = K.sum(weighted_f1)

    
    return weighted_f1 #for metrics, return only 'weighted_f1'

def weighted_f1(true_y, pred_y): #shapes (batch, 4)
#Daniel Moller - StackOverflow (o with umlaut above)
    #for metrics include these two lines, for loss, don't include them
    #these are meant to round 'pred' to exactly zeros and ones
    predLabels = K.argmax(pred_y, axis=-1)
    pred_y = K.one_hot(predLabels, 12) 


    ground_positives = K.sum(true_y, axis=0) + K.epsilon()       # = TP + FN
    pred_positives = K.sum(pred_y, axis=0) + K.epsilon()         # = TP + FP
    true_positives = K.sum(true_y * pred_y, axis=0) + K.epsilon()  # = TP
        #all with shape (4,)
    
    precision = true_positives / pred_positives 
    recall = true_positives / ground_positives
        #both = 1 if ground_positives == 0 or pred_positives == 0
        #shape (4,)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        #still with shape (4,)

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives) 
    weighted_f1 = K.sum(weighted_f1)

    
    return weighted_f1 #for metrics, return only 'weighted_f1'

@tf.function
def macroSoftF1(y, y_hat, from_logits = True):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)

    
    if from_logits == True:
        y = tf.nn.softmax(y)
        y_hat = tf.nn.softmax(y_hat)
    
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macroCost = tf.reduce_mean(cost) # average on all labels
    return macroCost

#%%
def get_group(traindf, group):
    """
    Stratified Group K-fold cross validation requires an extra group around which to split the data.
    This function returns that group data as an array. 
    
    Inputs:
    - traindf: str, the metadata CSV containing annotations for the images. 
    
    - group: str, set which location margin to divide the dataset by for cross validation.
    Options: 
        - "CTloc" - K-fold will split data down to singular CT stations. This is the smallest split of data.
        - "CTtype" - K-fold will split data down to CT study area and area type, three per CT area. ie. PT_RAF, PT_TRV, PT_ROW.
        - "areatype" - K-fold will split data down to area type, three overall. ie. RAF, TRV, ROW.
        - "studyarea" - K-fold will split data down to CT study area. This is the largest split of all data. ie. PT, SC, SV, GL.
    
    Returns:
        out (numpy.ndarray): array-like of shape [n_samples,]
        
    """
    
    train_df = pd.read_csv(traindf)

    if group in ("CTloc"):
        CTloc_train = np.array(train_df['Camera_loc'])
        return CTloc_train
    elif group in ("CTtype"):
        CTtype_train = np.array(train_df['Camera_type'])
        return CTtype_train
    elif group in ("Ecoregion"):
        Ecoregion_train = np.array(train_df['Ecoregion'])
        return Ecoregion_train
    elif group in ("studyarea"):
        studyarea_train = np.array(train_df['study_area'])
        return studyarea_train
    
#%%
#Construct DenseNet201 base model
class DenseNet201model(keras_tuner.HyperModel):
    def __init__(self, nclass, dataset):
        self._nclass = nclass
        self._dataset = dataset
        
    def build(self, hp):
        model = densenet.DenseNet201(weights="imagenet", include_top=False, input_shape=(224,224,3), pooling="avg")
        model.trainable = False    
        regularizer = None
        out = kl.Dense(256, activation="relu", kernel_regularizer=regularizer)(model.layers[-1].output)
       # out = kl.Dense(hp.Choice('Dense1Units', values=[128, 256]),
        #                   activation='relu', 
         #                  kernel_regularizer=regularizer)(model.layers[-1].output)
    
      #      out = kl.Dropout(hp.Float("Dropout1Value", min_value=0.05, max_value=0.5, step=0.05))(out)
        out = kl.Dropout(0.0)(out)
    
        out = kl.Dense(256, activation='relu')(out)
      #      out = kl.Dense(256, activation='relu')(out)
    
        #    out = kl.Dropout(hp.Float("Dropout2Value", min_value=0.05, max_value=0.5, step=0.05))(out)
        out = kl.Dropout(0.0)(out)
    
        spp_class = kl.Dense(self._nclass, activation='softmax', name="spp_class")(out)
    
        model = tf.keras.Model(inputs=model.inputs,
                               outputs=spp_class)
    
    #    lr=0.002
        lr=hp.Float("lr", min_value=0.0001, max_value=0.1, sampling="log")
        opt=Adam(learning_rate=lr)
        loss="categorical_crossentropy"
      #  loss=macroSoftF1
      
        if self._dataset in ("AHC"):
            metrics=["accuracy",
                  tf.keras.metrics.Precision(name="precision"),
                  tf.keras.metrics.Recall(name="recall"),
                  weighted_F1]
        elif self._dataset in ("MNRF"):
            metrics=["accuracy",
                  tf.keras.metrics.Precision(name="precision"),
                  tf.keras.metrics.Recall(name="recall"),
                  weighted_f1]

    
        model.compile(optimizer=opt,
                      loss=loss,
                      metrics=metrics)
        return model

#%%
#Construct DenseNet201 DANN model
class DANNseNet201model(keras_tuner.HyperModel):
    def __init__(self, nclass, dataset):
        self._nclass = nclass
        self._dataset = dataset
        
    def build(self, hp):
        model = densenet.DenseNet201(weights="imagenet", include_top=False, input_shape=(224,224,3), pooling="avg")
    #    model.trainable = True
        model.trainable = False    
    
    #    regularizer = tf.keras.regularizers.L1((hp.Float("L1reg", min_value=0.0, max_value=0.01, step=0.0005)))
    #    regularizer = tf.keras.regularizers.L2((hp.Float("L2reg", min_value=0.0, max_value=0.01, step=0.0005)))
        regularizer = None
    
    #    out = kl.Dense(hp.Choice('Dense1Units', values=[128, 256, 512]),
    #                       activation='relu', 
    #                       kernel_regularizer=regularizer)(model.layers[-1].output)
        out = kl.Dense(256, activation='relu', kernel_regularizer=regularizer)(model.layers[-1].output)
    #    out = kl.Dropout(hp.Float("Dropout1Value", min_value=0.05, max_value=0.5, step=0.05))(out)
        out = kl.Dropout(0.0)(out)
    
    #    out = kl.Dense(hp.Choice('Dense2Units', values=[64, 128, 256]), activation='relu')(out)
        out = kl.Dense(256, activation='relu', kernel_regularizer=regularizer)(out)
    #    out = kl.Dropout(hp.Float("Dropout2Value", min_value=0.05, max_value=0.5, step=0.05))(out)
        out = kl.Dropout(0.0)(out)
    
        #species output
        spp_class = kl.Dense(self._nclass, activation='softmax', name="spp_class")(out)
    
        #sincos domain output
        ReverseGrad = ReversalLayer()(out)
        domain = kl.Dense(2, activation="tanh", name="domain")(ReverseGrad)
    #    out_domain = kl.Dense(3, activation="softmax", name="out_domain")(ReverseGrad)
    
        model = tf.keras.Model(inputs=model.inputs,
    #                           outputs=[out_class, sine, cosine])
                               outputs=[spp_class, domain])
    
    #    lr=hp.Choice("lr", values=[0.0023, 0.0066, 0.0095, 0.01247, 0.0139])
        lr=0.0109
        opt=Adam(learning_rate=lr)
    
        losses = {"spp_class": 'categorical_crossentropy', 
                  "domain": "mean_squared_error"}
    #    losses = {"out_class": 'categorical_crossentropy', "out_domain": "cosine_similarity"}
    
        randomlossWeights = {"spp_class": 1, 
                             "domain":  0.1}
    
        metrics=["accuracy", weighted_F1] #,
    #              tf.keras.metrics.Precision(name="precision"),
     #             tf.keras.metrics.Recall(name="recall")]
        if self._dataset in ("AHC"):
            DANNmetrics={"spp_class":["accuracy", 
                                      tf.keras.metrics.Precision(name="precision"), 
                                      tf.keras.metrics.Recall(name="recall"), weighted_F1], 
                         "domain":"accuracy"}
        elif self._dataset in ("MNRF"):
            DANNmetrics={"spp_class":["accuracy", 
                                      tf.keras.metrics.Precision(name="precision"), 
                                      tf.keras.metrics.Recall(name="recall"), weighted_f1], 
                         "domain":"accuracy"}
    
        model.compile(optimizer=opt,
                      loss=losses,
                      loss_weights=randomlossWeights,
                      metrics=DANNmetrics)
        return model

#%%
#Construct DenseNet201 DANN model
class catDANNseNet201model(keras_tuner.HyperModel):
    def __init__(self, nclass, dataset):
        self._nclass = nclass
        self._dataset = dataset
        
    def build(self, hp):
        model = densenet.DenseNet201(weights="imagenet", include_top=False, input_shape=(224,224,3), pooling="avg")
    #    model.trainable = True
        model.trainable = False    
    
    #    regularizer = tf.keras.regularizers.L1((hp.Float("L1reg", min_value=0.0, max_value=0.01, step=0.0005)))
    #    regularizer = tf.keras.regularizers.L2((hp.Float("L2reg", min_value=0.0, max_value=0.01, step=0.0005)))
        regularizer = None
    
    #    out = kl.Dense(hp.Choice('Dense1Units', values=[128, 256, 512]),
    #                       activation='relu', 
    #                       kernel_regularizer=regularizer)(model.layers[-1].output)
        out = kl.Dense(256, activation='relu', kernel_regularizer=regularizer)(model.layers[-1].output)
    #    out = kl.Dropout(hp.Float("Dropout1Value", min_value=0.05, max_value=0.5, step=0.05))(out)
        out = kl.Dropout(0.0)(out)
    
    #    out = kl.Dense(hp.Choice('Dense2Units', values=[64, 128, 256]), activation='relu')(out)
        out = kl.Dense(256, activation='relu', kernel_regularizer=regularizer)(out)
    #    out = kl.Dropout(hp.Float("Dropout2Value", min_value=0.05, max_value=0.5, step=0.05))(out)
        out = kl.Dropout(0.0)(out)
    
        #species output
        spp_class = kl.Dense(self._nclass, activation='softmax', name="spp_class")(out)
    
        #categorical domain output
        ReverseGrad = ReversalLayer()(out)
        domain = kl.Dense(2, activation="softmax", name="domain")(ReverseGrad)
    #    out_domain = kl.Dense(3, activation="softmax", name="out_domain")(ReverseGrad)
    
        model = tf.keras.Model(inputs=model.inputs,
    #                           outputs=[out_class, sine, cosine])
                               outputs=[spp_class, domain])
    
    #    lr=hp.Choice("lr", values=[0.0023, 0.0066, 0.0095, 0.01247, 0.0139])
        lr=0.0109
        opt=Adam(learning_rate=lr)
    
        losses = {"class": 'categorical_crossentropy', 
                  "domain": "categorical_crossentropy"}
    #    losses = {"class": 'categorical_crossentropy', 
    #              "domain": "mean_squared_error"}
    
        randomlossWeights = {"spp_class": 1, 
                             "domain":  0.1}
    
        if self._dataset in ("AHC"):
            DANNmetrics={"spp_class":["accuracy", 
                                      tf.keras.metrics.Precision(name="precision"), 
                                      tf.keras.metrics.Recall(name="recall"), weighted_F1], 
                         "domain":["accuracy", 
                                       tf.keras.metrics.Precision(name="precision"), 
                                       tf.keras.metrics.Recall(name="recall"), cat_wf1]}
        elif self._dataset in ("MNRF"):
            DANNmetrics={"spp_class":["accuracy", 
                                      tf.keras.metrics.Precision(name="precision"), 
                                      tf.keras.metrics.Recall(name="recall"), weighted_f1], 
                         "domain":["accuracy", 
                                       tf.keras.metrics.Precision(name="precision"), 
                                       tf.keras.metrics.Recall(name="recall"), cat_wf1]}
               
        model.compile(optimizer=opt,
                      loss=losses,
                      loss_weights=randomlossWeights,
                      metrics=DANNmetrics)
        return model
#%% Command-line driver

def strbool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in ('true', 't', 'yes', 'y'):
        return True
    elif string.lower() in ('false', 'f', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected. Enter either "True" or "False".')

def main():
    
    parser = argparse.ArgumentParser(
        description='This program generates batches of data for training and testing the ML model.')
    parser.add_argument('traindf', 
                        type=str,
                        help='Path to df containing list of images with metadata.')
    parser.add_argument('dataset', 
                        type=str,
                        help='Which dataset would you like to use?'+ \
                            'Options: "AHC" or "MNRF.')
    parser.add_argument('hypmodel', 
                        type=str,
                        help='Which model would you like to use?'+ \
                            'Options: "DenseNet201", "DANNseNet201", "catDANN".')
    parser.add_argument('save', 
                        type=str,
                        help='What would you like to name the save file?')
    parser.add_argument('nested',
                        type=strbool,
                        default=True,
                        help='Str, enter "True" or "t" or "False" or "f" if Cross-validation should be nested. ' +\
                            'Argument can be either upper or lower case. ' +\
                            'Default is False.')
#    parser.add_argument('cv', 
#                        type=str,
#                        help='What type of cross validation would you like to use?'+ \
#                            'Options: "StratifiedKFold" ("skf") or "StratifiedGroupKFold" ("sgkf").')
    parser.add_argument('--nfold', 
                        type=int,
                        default=5,
                        help='How many splits would you like to use?'+ \
                            'Default is 5.')
    parser.add_argument('--group', 
                        type=str,
                        default=None,
                        help='Would you like to group split by location or camera type?')
    parser.add_argument('--batch_size', '--batch',
                        type=int, 
                        default=64,
                        help='Integer; number of images in batch for training. Default is 50. ')
    parser.add_argument('--epochs',
                        type=int, 
                        default=50,
                        help='Integer; number of eopchs to run per trial. Default is 50. ')
    parser.add_argument('--trials',
                        type=int, 
                        default=10,
                        help='Integer; number of trials to run. Default is 10. ')
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
    
    args = parser.parse_args()
    
    assert os.path.exists(args.traindf), \
        'df {} does not exist'.format(args.traindf)        
#    if args.group is None and args.cv in ('StratifiedGroupKFold', 'sgkf', 'SGKF'):
#        raise ValueError("Grouping argument must be included if using the StratifiedGroupKFold CV Tuner.")
#    assert os.path.exists(args.testdf), \
#        'df {} does not exist'.format(args.testdf)

    traindf = pd.read_csv(args.traindf)
    if args.dataset in ("AHC"):
        class_names = ['moose', 'fox', 'deer', 'sandhill crane', 'bear', 'domestic dog']
        nclass = 6
    elif args.dataset in ("MNRF"):
        class_names = ['snowshoe_hare', 'American black_bear', 'moose', 
                      'red_squirrel', 'white-tailed deer', 'ruffed_grouse', 
                      'e_chipmunk', 'wolf', 'turkey vulture', 'American marten', 
                      'Peromyscus spp', 'fisher']
        nclass = 12
    seasons = ['fall', 'winter', 'spring', 'summer']

    if args.group:
        group = get_group(args.traindf, args.group)
        kfold = StratifiedGroupKFold(n_splits=args.nfold, shuffle=True, random_state=42)
        name = ("{} Group").format(args.group)
        group_name = args.group
    else:
        group = None
        kfold = StratifiedKFold(n_splits=args.nfold, shuffle=True, random_state=42)
        name = "Random"
        group_name = "CTRand"
    
#    if args.cv in ('StratifiedKFold', 'SKF', 'skf'):
#        kfold = StratifiedKFold(n_splits=args.nfold, shuffle=True, random_state=42)
#        name = "Random"
#    elif args.cv in ('StratifiedGroupKFold', 'sgkf', 'SGKF'):
#        kfold = StratifiedGroupKFold(n_splits=args.nfold, shuffle=True, random_state=42)
#        name = ("{} Group").format(args.group)

    print("Training data and options extracted. Preparing to load model.")
#keras_tuner.HyperParameters()
    if args.hypmodel in ('DenseNet201', 'CNN'):
        print("Loading DenseNet201 model.")
        load_time = time.time()
        model=DenseNet201model(nclass=nclass, dataset=args.dataset)
        if args.nested==True:
            from SKFTuners import Nested_SKF_Tuner
            tuner = Nested_SKF_Tuner(BayesianOptimization)(model,
                                                 kfoldcv = kfold,
                                                 group = group,
                                                 df=traindf,
                                                 dataset=args.dataset,
                                                 model_name = args.hypmodel,
                                                 class_names = class_names,
                                                 domain_names=None,
                                                 group_name=group_name,
                                                 batch_size=args.batch_size,
                                                 objective=Objective("val_accuracy", direction="max"),
                                                 max_trials=args.trials,
                                                 save_output=False,
                                                 save_history=True,
                                                 directory="Results",
                                                 project_name=args.save,
                                                 seed=42,
                                                 overwrite=True,)
            elapsed = time.time() - load_time
            print(("Model loaded in {}. Nested Stratified {} K-Fold Tuner saved with DenseNet201.").format(elapsed, name))
        else:
            from SKFTuners import Outer_SKF_Tuner
            tuner = Outer_SKF_Tuner(kfoldcv = kfold,
                                    tuner_class = BayesianOptimization,
                                    hypermodel = DenseNet201model,
                                    group = group,
                                    df = traindf,
                                    objective=Objective("val_accuracy", direction="max"),
                                    max_trials=args.trials,
                                    directory="Results",
                                    project_name=args.save,
                                    seed=42,
                                    overwrite=True)
            elapsed = time.time() - load_time
            print(("Model loaded in {}. Unnested Stratified {} K-Fold Tuner saved with DenseNet201.").format(elapsed, name))
    elif args.hypmodel in ('DANNseNet201', 'DANN'):
        load_time = time.time()
        model=DANNseNet201model(nclass=nclass, dataset=args.dataset)
        if args.nested==True:
            from SKFTuners import Nested_SKF_Tuner
            tuner = Nested_SKF_Tuner(BayesianOptimization)(hypermodel = model,
                                                 kfoldcv = kfold,
                                                 group = group,
                                                 df=traindf,
                                                 dataset=args.dataset,
                                                 model_name = args.hypmodel,
                                                 class_names = class_names,
                                                 domain_names=seasons,
                                                 group_name=group_name,
                                                 batch_size=args.batch_size,
                                                 objective=Objective("val_spp_class_accuracy", direction="max"),
                                                 max_trials=args.trials,
                                                 save_output=False,
                                                 save_history=True,
                                                 save_domvec=True,
                                                 directory="Results",
                                                 project_name=args.save,
                                                 seed=42,)
            elapsed = time.time() - load_time
            print(("Model loaded in {}. Nested Stratified {} K-Fold Tuner saved with DANNseNet201.").format(elapsed, name))
        else:
            from SKFTuners import Outer_SKF_Tuner
            tuner = Outer_SKF_Tuner(kfoldcv = kfold,
                                    tuner_class = BayesianOptimization,
                                    hypermodel = DANNseNet201model,
                                    group = group,
                                    df = traindf,
                                    objective=Objective("val_spp_class_accuracy", direction="max"),
                                    max_trials=args.trials,
                                    directory="Results",
                                    project_name=args.save,
                                    seed=42,
                                    overwrite=True)
            elapsed = time.time() - load_time
            print(("Model loaded in {}. Unnested Stratified {} K-Fold Tuner saved with DANNseNet201.").format(elapsed, name))
    elif args.hypmodel in ('CatDANN', 'catDANN'):
        load_time = time.time()
        model = catDANNseNet201model(nclass=nclass, dataset=args.dataset)
        if args.nested==True:
            from SKFTuners import Nested_SKF_Tuner
            tuner = Nested_SKF_Tuner(BayesianOptimization)(hypermodel = model,
                                                 kfoldcv = kfold,
                                                 group = group,
                                                 df=traindf,
                                                 dataset=args.dataset,
                                                 model_name = args.hypmodel,
                                                 class_names = class_names,
                                                 domain_names=seasons,
                                                 group_name=group_name,
                                                 batch_size=args.batch_size,
                                                 objective=Objective("val_spp_class_accuracy", direction="max"),
                                                 max_trials=args.trials,
                                                 save_output=False,
                                                 save_history=True,
                                                 save_domvec=True,
                                                 directory="Results",
                                                 project_name=args.save,
                                                 seed=42,)
            elapsed = time.time() - load_time
            print(("Model loaded in {}. Nested Stratified {} K-Fold Tuner saved with SnowNet201.").format(elapsed, name))
    else:
        raise argparse.ArgumentTypeError('Model string expected. Please enter desired model. ' + \
                                         'Module is currently optimised for "DenseNet201" only.')
    
    
    #Define EarlyStopping callback.
    if args.hypmodel in ('DANNseNet201', 'CatDANN', 'catDANN'):
        es = EarlyStopping(monitor='val_spp_class_loss', mode='min', patience=15)
    else:
        es = EarlyStopping(monitor='val_loss', mode='min', patience=15)

    start_time = time.time()  
    
    # Run the Hyperparameter Random Search
    print("Beginning hyperparameter random search.")
    if args.hypmodel in ('DenseNet201'):
        tuner.search(traindf, class_names=class_names, validation_split=0.2,
                     batch_size=args.batch_size, epochs=args.epochs, 
                     callbacks=[es], verbose=2)
    elif args.hypmodel in ('DANNseNet201'):
        tuner.search(traindf, class_names=class_names, validation_split=0.2, 
                     batch_size=args.batch_size, epochs=args.epochs, 
                     callbacks=[es], verbose=2)
    elif args.hypmodel in ('CatDANN', 'catDANN'):
        tuner.search(traindf, class_names=class_names, domain_names = seasons, 
                     validation_split=0.2, batch_size=args.batch_size,
                     epochs=args.epochs, callbacks=[es], verbose=2)

    elapsed = time.time() - start_time
    
#    if args.nested==True:
    df = pd_inner_cv_get_result(tuner)
    df.to_csv("Results/" + args.hypmodel + '_' + group_name + '_' + args.save + ".csv")
#    else:
 #       print(tuner.get_best_hparams())
  #      tuner.get_best_hparams
   #     print(tuner.get_best_hparams)
    #    print(tuner.get_best_hparams())
     #   eval = tuner.evaluate(X_train, y_train, validation_split=0.2, 
      #                        batch_size=args.batch_size, validation_batch_size=args.batch_size, 
       #                       epochs=args.epochs, callbacks=[es],  verbose=2)
       # print(eval)
    
    print(('Finished running search in {}.').format(humanfriendly.format_timespan(elapsed)))   
  #  models = tuner.get_best_models(num_models=5)
   # best_model = models[0]
    #print(best_model)
              
    print("="*40)
    print('Random Search complete.')
       
    print("Goodbye.")
    print("And good luck!")

if __name__ == '__main__':
    main()

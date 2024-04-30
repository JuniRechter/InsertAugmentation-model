# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:22:36 2023

@author: hayle
"""
import copy
import warnings
import os
import json

import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import sklearn.metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
import ImageDataGenerators as idg
from keras_tuner.engine.tuner import Tuner, maybe_distribute

from keras_tuner import utils
from keras_tuner.engine import tuner_utils
from keras_tuner.engine import trial as trial_module
from keras_tuner_cv.utils import get_metrics_std_dict

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import itertools
from sklearn.utils import resample
import seaborn as sns
from multiprocessing import Process
#%%
light_palette = ["#221330", "#390962", "#5F126E", 
           "#85206A", "#A92E5E", "#CB4049", 
           "#E65C2E", "#FBAE12", "#F4DB4B", "#FFF999"]

dark_palette =["#FFF999", "#F4DB4B", "#FBAE12", "#E65C2E",
               "#CB4049", "#A92E5E", "#85206A",
               "#5F126E", "#390962", "#221330"]

#%%
def get_group(train_df, group):
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
    
    if group in ("CTloc"):
        CTloc_train = np.array(train_df['Camera_loc'])
        return CTloc_train
    elif group in ("CT_loc"):
        CTtype_train = np.array(train_df['CT_location'])
        return CTtype_train
    elif group in ("Ecoregion"):
        Ecoregion_train = np.array(train_df['Ecoregion'])
        return Ecoregion_train
    elif group in ("studyarea"):
        studyarea_train = np.array(train_df['study_area'])
        return studyarea_train
#%%
def get_stats(df, column):
    for cat in df[column].unique():
        df_length = len(df)
        spp_length = len(df[df[column] == cat])
        average=(spp_length/df_length)
        average=("%.3f" % average)
        print(column + ": " + cat + ", Count: " + str(spp_length) + ", Average: " + str(average))
#%%
def upsample_data(train_df, MAX_SAMPLE=80000):
    
    train_sample = pd.DataFrame()

    temp = resample(train_df, 
                    replace=True,           # sample with replacement
                    n_samples=MAX_SAMPLE,          # to match majority class
                    random_state=42,        # reproducible results
                    stratify=train_df['path']) #must sample unique items before replacement
    
            # Combine majority class with upsampled minority class
    train_sample = pd.concat([train_sample, temp])
 
    return train_sample
#%%
def get_empties(train_df, train_empties, MAX_SAMPLE=80000, LOC_BAL=False, NIGHT_BAL=False):
    
    if NIGHT_BAL == True:
        n_loc = train_empties.Loc_ToD.nunique()
        column="Loc_ToD"
    else:
        n_loc = train_empties.Camera_loc.nunique()
        column="Camera_loc"
    sample_size = int(MAX_SAMPLE/n_loc)
    
    empty_sample = pd.DataFrame()

    if LOC_BAL == True:
        for cat in train_empties[column].unique():
            empties_df = train_empties[train_empties[column] == cat]
            df = train_df[train_df[column]==cat]
            CT_sample = sample_size-len(df)
            if CT_sample <=1:
                CT_sample=1
            temp = resample(empties_df, 
                            replace=True,           # sample with replacement
                            n_samples=CT_sample,          # to match majority class
                            random_state=42,        # reproducible results
                            stratify=empties_df['path']) #must sample unique items before replacement
    
            # Combine majority class with upsampled minority class
            empty_sample = pd.concat([empty_sample, temp])
    else:
        if NIGHT_BAL == True:
            for cat in train_empties['Night'].unique():
                empties_df = train_empties[train_empties['Night'] == cat]
                CT_sample = int((MAX_SAMPLE-len(train_df))/2)
                if CT_sample <=1:
                    CT_sample=1
                
                temp = resample(empties_df, 
                                replace=True,           # sample with replacement
                                n_samples=CT_sample,          # to match majority class
                                random_state=42,        # reproducible results
                                stratify=empties_df['path']) #must sample unique items before replacement

            # Combine majority class with upsampled minority class
            empty_sample = pd.concat([empty_sample, temp])
            
        else:
            empties_df = train_empties
            CT_sample = int(MAX_SAMPLE-len(train_df))
            if CT_sample <=1:
                CT_sample=1

            temp = resample(empties_df, 
                            replace=True,           # sample with replacement
                            n_samples=CT_sample,          # to match majority class
                            random_state=42,        # reproducible results
                            stratify=empties_df['path']) #must sample unique items before replacement

        # Combine majority class with upsampled minority class
        empty_sample = pd.concat([empty_sample, temp])
 
    return empty_sample
#%%
def get_train_weights(train_df, empties_length, SPP_BAL=True):
    class_weights = []
    night_weights=[]
    n_spp = train_df.id.nunique()
    if SPP_BAL==True:
        max_id_count = empties_length/n_spp
        for i in range(n_spp):
            class_weights.append((max_id_count - len(train_df.loc[train_df['id']==i]))/empties_length)
            if i !=5:
                night_weights.append((max_id_count - len(train_df.loc[train_df['id']==i]))/empties_length)
    else:
        for i in range(n_spp):
            class_weights.append((len(train_df.loc[train_df['id']==i]))/len(train_df))
            if i !=5:
                night_weights.append((len(train_df.loc[train_df['id']==i]))/len(train_df))

    return class_weights, night_weights
#%%
def generate_confusion_matrix(cnf_matrix, classes, colour='light'):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    np.set_printoptions(precision=2)
    figure = plt.figure(figsize=(12,10))

    normcm= cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

    if colour == 'light':
        inferno = sns.blend_palette(light_palette, n_colors=10, as_cmap=True)
    else:
        inferno = sns.blend_palette(dark_palette, n_colors=10, as_cmap=True)

    plt.imshow(normcm, interpolation='nearest', cmap=plt.get_cmap(inferno))

    plt.colorbar()
    plt.clim(0,1)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=13, rotation=45)
    plt.yticks(tick_marks, classes, fontsize=13, rotation=45)

    normfmt = '.1%' 
    fmt = 'd'

    thresh = 0.585

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        if colour =='light':
            text="black" if normcm[i, j] > (thresh) else "white"
        else:
            text="white" if normcm[i, j] > (thresh) else "black"

        plt.text(j, i, format(normcm[i, j], normfmt), horizontalalignment="center",
                 verticalalignment="top", color=text, fontsize=15)
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 verticalalignment="bottom", color=text, fontweight="bold", fontsize=15)

    #  plt.tight_layout()
    plt.ylabel('True label', fontweight="bold", fontsize=15)
    plt.xlabel('Predicted label', fontweight="bold", fontsize=15)

    return figure
#%%
#Generate_df
def generate_df(path, class_true, class_pred, classes, DANN=False, model=None, dom_true=None, dom_pred=None):
    
    df=pd.DataFrame()
    image_path=path.tolist()
    df['image_path']=image_path
    
    class_dict = dict(enumerate(classes))
    
    true_class=(class_true.astype(int)).tolist()
    df['true_class']=pd.DataFrame(true_class)
    df['true_animal'] = df['true_class'].apply(lambda x: class_dict[x])

    pred_class=(class_pred.astype(int)).tolist()
    df['pred_class']=pd.DataFrame(pred_class)
    df.to_csv('Results/imagepreds.csv')
    df['pred_animal'] = df['pred_class'].apply(lambda x: class_dict[x])
    
    if DANN==True:
        if dom_true is None:
            print("Error: dom_true was not provided for df." + \
                  "Returning df with class predictions only.")
            return df
        elif dom_pred is None:
            print("Error: dom_pred was not provided for df." + \
                  "Returning df with class predictions only.")
            return df
        if model in ('DANNseNet201', 'DANN'):
            true_dom=dom_true.tolist()
            df[['true_sine', 'true_cosine']] = pd.DataFrame(true_dom)

            df['true_DoY']=((np.arctan2(df.true_sine, df.true_cosine))*365)/(2*np.pi)
            df['true_DoY'] = [(365 + ele) if ele <0 else ele for ele in df['true_DoY']]
            df['true_DoY'] = [(15 + ele) if ele <=350 else (ele + 15 - 365) for ele in df['true_DoY']]

            df['true_angle']=((np.arctan2(df.true_sine, df.true_cosine))*180)/(np.pi)
            df['true_angle'] = [(180 + ele) if ele <0 else ele for ele in df['true_angle']]

            pred_dom=dom_pred.tolist()
            df[['pred_sine', 'pred_cosine']] = pd.DataFrame(pred_dom)
#            df['pred_sine'] =df['pred_sine']/0.8
 #           df['pred_cosine'] =df['pred_cosine']/0.8
            df['pred_DoY']=((np.arctan2(df.pred_sine, df.pred_cosine))*365)/(2*np.pi)
            df['pred_DoY'] = [(365 + ele) if ele <0 else ele for ele in df['pred_DoY']]
            df['pred_DoY'] = [(15 + ele) if ele <=350 else (ele + 15 - 365) for ele in df['pred_DoY']]

            df['pred_angle']=((np.arctan2(df.pred_sine, df.pred_cosine))*180)/(np.pi)
            df['pred_angle'] = [(180 + ele) if ele <0 else ele for ele in df['pred_angle']]

            df["true_season"] = pd.cut(df['true_DoY'],
                                       bins=[1, 60, 152, 244, 335, 366],
                                       labels=["Winter", "Spring", "Summer", "Fall", "Winter"],
                                       ordered=False)

            df["pred_season"] = pd.cut(df['pred_DoY'],
                                       bins=[1, 60, 152, 244, 335, 366],
                                       labels=["Winter", "Spring", "Summer", "Fall", "Winter"],
                                       ordered=False)
        elif model in ('catDANN', 'CatDANN'):
            seasons = ['fall', 'winter', 'spring', 'summer']
            season_dict = dict(enumerate(seasons))
            true_dom=(dom_true.astype(int)).tolist()
            df['true_domain']=pd.DataFrame(dom_true)
            df['true_season'] = df['true_domain'].apply(lambda x: season_dict[x])
            pred_dom=(dom_pred.astype(int)).tolist()
            df['pred_domain']=pd.DataFrame(pred_dom)
            df['pred_season'] = df['pred_domain'].apply(lambda x: season_dict[x])

    return df

#%%
def generate_domain_truepred(df):
    """
    Returns a matplotlib figure containing the plotted domain labels and predictions.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    predsin=df['pred_sine']
    predcos=df['pred_cosine']
    
    truesin=df['true_sine']
    truecos=df['true_cosine']
    
    #np.set_printoptions(precision=2)
    figure = plt.figure(figsize=(12,12))
    axis = figure.add_subplot(111)
    axis.grid(which="both", color='grey', linestyle='dotted', alpha=0.5)
    
    for i in range(len(df)):
        plt.plot([truesin[i],predsin[i]], [truecos[i], predcos[i]], alpha=0.5)
    
    axis.scatter(x=truesin,y=truecos, c='#62146E', #Purple
                 marker="D", s=120, alpha=0.5, label="True labels")
    axis.scatter(x=predsin,y=predcos, c='#F16F20',  #Gold
                 marker="o", s=100, alpha=0.5, label="Predicted labels")

    axis.tick_params(which='major', length=9, labelsize=14)
    axis.tick_params(which='minor', length=7)
    axis.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axis.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))

    plt.annotate(text="Fall",xy=[-1.0, -0.52], fontsize=15)
    plt.annotate(text="Winter",xy=[-0.57, 0.92], fontsize=15)
    plt.annotate(text="Spring",xy=[0.91, 0.48], fontsize=15)
    plt.annotate(text="Summer",xy=[0.4, -1.0], fontsize=15)

    plt.ylabel('Cosine Normalised Time', labelpad=0.5, fontsize=17) #fontweight="bold")
    plt.xlabel('Sine Normalised Time', fontsize=17) #fontweight="bold")
    axis.legend(fontsize=15)
    return figure
#%%
def generate_doy_truepred(df): #, legend="out"):
    figure = plt.figure(figsize=(12,12))
    axis = figure.add_subplot(111)
    cdict = {"Fall": '#E30513', "Winter": '#78D2EB', "Spring": '#62D658', "Summer": '#D68802'} #COLOURBLIND SAFE PALETTE
    grouped = df.groupby("true_season", observed=True)

    for key, group in grouped:
        group.plot(ax=axis, kind='scatter', x='true_DoY', y='pred_DoY', 
                   label=key, color=cdict[key], marker="D", s=100, alpha=0.65)

    axis.grid(which="minor", color='grey', linestyle='dotted', alpha=0.3)
    axis.grid(which="major", color='grey', linestyle='dotted', alpha=0.5)
    
    axis.tick_params(which='major', length=9, labelsize=12)
    axis.tick_params(which='minor', length=7)
    axis.set_xticks([0, 90, 180, 270, 360])
    axis.set_yticks([0, 90, 180, 270, 360])
    axis.set_xlim([0,366])
    axis.set_ylim([0,366])
    axis.xaxis.set_minor_locator(AutoMinorLocator(3))
    axis.yaxis.set_minor_locator(AutoMinorLocator(3))
    
    plt.ylabel('Predicted Day of Year', labelpad=0.5, fontsize=17) #fontweight="bold")
    plt.xlabel('True Day of Year', fontsize=17) #fontweight="bold")
    
    winter = plt.Line2D([0],[0], label='Winter', color= '#78D2EB', 
                        marker='D', markersize=10, alpha=0.65, linestyle='')
    spring = plt.Line2D([0],[0], label='Spring', color= '#62D658', 
                        marker='D', markersize=10, alpha=0.65, linestyle='')
    summer = plt.Line2D([0],[0], label='Summer', color= '#D68802', 
                        marker='D', markersize=10, alpha=0.65, linestyle='')
    fall = plt.Line2D([0],[0], label='Fall', color= '#E30513', 
                      marker='D', markersize=10, alpha=0.65, linestyle='')

    handles=[winter, spring, summer, fall]
#    if legend=="out":
 #       axis.legend(handles=handles, bbox_to_anchor=(1.2, 0.593), title="True Season", title_fontsize=15, fontsize=15)
  #  else:
    axis.legend(handles=handles, title="True Season", title_fontsize=15, fontsize=15) #bbox_to_anchor=(1.2, 1.01), 
    return figure
#%%
def Nested_SKF_Tuner(superclass: Tuner,):
    class SKF_CV(superclass):
        """
        Hyparameters search evaluated using Stratified Cross-Validation over a
        parameter space.
        """

        def __init__(self,
                     hypermodel,
                     kfoldcv: StratifiedKFold,
                     df,
                     crops:None,
                     empties:None,
                     model_name,
                     class_names,
                     domain_names:None,
                     group_name,
                     *args,
                     balance:None,
                     save_history=False,
                     save_output=False,
                     save_cm=True,
                     save_domvec=False,
                     restore_best=True,
                     batch_size,
                     INSERTS=False,
                     SPP_BAL,
                     NIGHT_BAL,
                     LOC_BAL,
                     UPSAMPLE,
                     ADD_UNSEEN,
                     MAX_SAMPLE,
                     **kwargs,):
            """Stratified CV Tuner constructor.

            Args:
                cv (BaseCrossValidator): instance of cross validator to use.
            """
            super(SKF_CV, self).__init__(hypermodel, *args, **kwargs)
            self._kfoldcv = kfoldcv
            self.df = df.copy()
            self.crops = crops.copy()
            self.empties = empties.copy()
            self._model = model_name
            self._class_names = class_names
            self._balance = balance
            self._domain_names = domain_names
            self._group_name = group_name
            self._save_history = save_history
            self._save_output = save_output
            self._save_cm = save_cm
            self._save_domvec = save_domvec
            self._restore_best = restore_best
            self._batch_size = batch_size
            self._verbose = 2
            self._INSERTS= INSERTS
            self._SPP_BAL = SPP_BAL
            self._NIGHT_BAL = NIGHT_BAL
            self._LOC_BAL = LOC_BAL
            self._UPSAMPLE = UPSAMPLE
            self._ADD_UNSEEN = ADD_UNSEEN
            self._MAX_SAMPLE = MAX_SAMPLE
            
        def search(self, *fit_args, **fit_kwargs):
            if "verbose" in fit_kwargs:
                self._verbose = fit_kwargs.get("verbose")
            self.on_search_begin()
            while True:
                trial = self.oracle.create_trial(self.tuner_id)
                if trial.status == trial_module.TrialStatus.STOPPED:
                    # Oracle triggered exit.
                    tf.get_logger().info("Oracle triggered exit")
                    break
                if trial.status == trial_module.TrialStatus.IDLE:
                    # Oracle is calculating, resend request.
                    continue

                self.on_trial_begin(trial)
                results = self.run_trial(trial, *fit_args, **fit_kwargs)
                # `results` is None indicates user updated oracle in `run_trial()`.
                if results is None:
                    warnings.warn("`Tuner.run_trial()` returned None. It should return one of "
                                  "float, dict, keras.callbacks.History, or a list of one "
                                  "of these types. The use case of calling "
                                  "`Tuner.oracle.update_trial()` in `Tuner.run_trial()` is "
                                  "deprecated, and will be removed in the future.",
                                  DeprecationWarning, stacklevel=2,)
                else:
                    metrics = tuner_utils.convert_to_metrics_dict(results, self.oracle.objective)
                    metrics.update(get_metrics_std_dict(results))
                    self.oracle.update_trial(trial.trial_id, metrics,)
                self.on_trial_end(trial)
            self.on_search_end()

        def run_trial(self, trial, class_names, *args, **kwargs):
            original_callbacks = kwargs.pop("callbacks", [])
            classes = class_names
            df = self.df
            group=None
            if self._group_name in ("CTRand"):
                traindf, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['id'])
                group = None
            elif self._group_name in ("CTloc"):
                splitter = GroupShuffleSplit(test_size=0.10, n_splits=2, random_state=4)
                Y = np.array(df['id'])
                testsplit = splitter.split(df, y=Y, groups=df['CT_location'])
                train_index, test_index = next(testsplit)
                traindf = df.iloc[train_index]
                group = get_group(traindf, "CT_loc")
                test_df = df.iloc[test_index]
                test_group = np.unique(group[test_index])

            Y = np.array(traindf['id'])
            num_classes = (df['id'].nunique())

            histories = []
            tot_predicted_y = np.array([])
            tot_true_y = np.array([])
            tot_predicted_dom = np.array([])
            tot_true_dom = np.array([])
            reports = []
            # Run the training process multiple times.            
            for execution in range(self.executions_per_trial):
                # Run the training over different splits.
                for split, (train_index, val_index) in enumerate(self._kfoldcv.split(traindf, Y, group)):
                    if self._verbose ==2: tf.get_logger().info("\n" + "-" * 30 + "\n"
                                                           f"K-fold Cross-Validation {split}/{self._kfoldcv.get_n_splits()}"
                                                           + "\n" + "-" * 30 + "\n")
                    print(f"Fold {split}:")

                    train_df = traindf.iloc[train_index]
                    val_df = traindf.iloc[val_index]

                    train_count=len(train_df)
                    val_count=len(val_df)
                    train_prop = round((train_count/(train_count+val_count)), 3)
                    val_prop = round((val_count/(train_count+val_count)), 3)
                    
                    if group is not None:
                      #  print("Training groups: ", flush=True)
                      #  print((np.unique(group[train_index])), flush=True)
                      #  print(f"    Train count: {train_count}, train percentage: {train_prop}.", flush=True)
                #                    print(train_df[['Species', 'id']].value_counts().reset_index(name='count'), flush=True)
                        train_group = np.unique(group[train_index])

                        print("\n" + "Validation groups: ", flush=True)
                        print((np.unique(group[val_index])), flush=True)
                        print(f"    Val count: {val_count}, val percentage: {val_prop}", flush=True)
                #                     print(val_df[['Species', 'id']].value_counts().reset_index(name='count'), flush=True)
                        val_group = np.unique(group[val_index])
                    if self._INSERTS==True:
                        if self._group_name in ("CTloc"):
                            train_empties=self.empties.loc[self.empties['CT_location'].isin(train_group)]
                            train_empties.loc[:,'id'] = -1
                            if self._ADD_UNSEEN ==True:
                                val_empties=self.empties.loc[self.empties['CT_location'].isin(val_group)]
                                test_empties=self.empties.loc[self.empties['CT_location'].isin(test_group)]
                                val_empties.loc[:,'id'] = -2
                                test_empties.loc[:,'id'] = -2
                                train_empties = pd.concat([train_empties, val_empties, test_empties])
                        else:
                            train_empties = self.empties.copy()
                        train_empties=get_empties(train_df, train_empties, MAX_SAMPLE=self._MAX_SAMPLE, 
                                                  LOC_BAL=self._LOC_BAL, NIGHT_BAL=self._NIGHT_BAL)
                        empties_length = len(train_empties)
                        if self._SPP_BAL==True:
                            empties_length = len(train_empties.loc[train_empties['id']==-1])
                            train_weights, night_weights = get_train_weights(train_df, empties_length, True)
                            print(train_weights)
                            print(night_weights)
                        else:
                            train_weights, night_weights = get_train_weights(train_df, empties_length, False)
                            print(train_weights)
                            print(night_weights)
                
                        add_empties = [train_df, train_empties]
                        train_df = pd.concat(add_empties)
                        print(len(train_df))
                    #                        print(len(train_df), flush=True)
                    else:
                        if self._UPSAMPLE==True:
                            print(train_df[['Species', 'id']].value_counts().reset_index(name='count'), flush=True)
                            print(train_df['path'].nunique())
                            train_df = upsample_data(train_df, self._MAX_SAMPLE)
                            print(train_df['path'].nunique())
                        train_weights = None
                        night_weights = None
                    print("\n" + "Training data length: ", flush=True)
                    print(len(train_df), flush=True)
                    print(train_df[['Species', 'id']].value_counts().reset_index(name='count'), flush=True)
                    
                    traingen = idg.TrainingDataGenerator(train_df, self.crops,
                                                         batch_size = self._batch_size,
                                                         y_col = {'id': 'id', 'domain': 'domain', 
                                                                  'sin_date': 'adjsin_date', 'cos_date': 'adjcos_date'},
                                                         X_col = {'path': 'path', 'Night':'Night'},
                                                         model_name=self._model, shuffle = True, 
                                                         INSERTS=self._INSERTS, SPP_BAL=self._SPP_BAL,
                                                         weights=train_weights, night_weights=night_weights)
                    valgen = idg.ValidationDataGenerator(val_df,
                                                         batch_size = self._batch_size,
                                                         y_col = {'id': 'id', 'domain': 'domain', 
                                                                  'sin_date': 'adjsin_date', 'cos_date': 'adjcos_date'},
                                                         X_col = {'path': 'path'},
                                                         model_name=self._model,
                                                         shuffle = True)
                    testgen = idg.ValidationDataGenerator(test_df,
                                                          batch_size=1,
                                                          y_col = {'id': 'id', 'domain': 'domain',
                                                                   'sin_date': 'adjsin_date', 'cos_date': 'adjcos_date'},
                                                          X_col = {'path': 'path'},
                                                          model_name=self._model,
                                                          shuffle = False)
                    predicted_y = np.array([])
                    true_y = np.array([])
                    # Create a copy of args and kwargs to fill with fold-specific data
                    copied_args = []
                    copied_kwargs = copy.copy(kwargs)

                    # Set the training set
                    copied_kwargs["x"] = traingen
                    # Get the validation set
                    test_paths = np.array(test_df["path"])
                    y_test = np.array(test_df['id'])
                    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

                    if self._model in ("DANNseNet201"):
                        sin_date = np.array(val_df['adjsin_date'])
                        cos_date = np.array(val_df['adjcos_date'])
                        sincos_val = np.asarray([y for y in zip(sin_date, cos_date)])
                    elif self._model in ('CatDANN', 'catDANN'):
                        predicted_dom = np.array([])
                        true_dom = np.array([])
                        cat_domain = np.array(val_df['domain'])
                        domain_val = tf.keras.utils.to_categorical(cat_domain, num_classes=4)

                    # Set the validation set
                    copied_kwargs["validation_data"] = valgen

                    # -------------------------------------------------------
                    # Callbacks
                    # -------------------------------------------------------

                    # Configure tensorboard
                    callbacks = self._deepcopy_callbacks(original_callbacks)
                    self._configure_tensorboard_dir(callbacks, trial, str(execution) + "_" + str(split))
                    model_checkpoint = tuner_utils.SaveBestEpoch(objective=self.oracle.objective,
                                                                 filepath=self._get_checkpoint_fname(trial.trial_id) + "_" + str(execution) + "_" + str(split),)
                    
                    callbacks.append(tuner_utils.TunerCallback(self, trial))
                    # Save all the checkpoints.
                    # The file name will be checkpoint_{execution}_{split}
                    callbacks.append(model_checkpoint)
                    copied_kwargs["callbacks"] = callbacks

                    # Build and train the model
                    history, model = self._build_and_fit_model(trial, *copied_args, **copied_kwargs)

                    if self._restore_best:
                        # Load the best epoch according to objective function
                        model = self._try_build(trial.hyperparameters)
                        model.load_weights(self._get_checkpoint_fname(trial.trial_id)
                            + "_" + str(execution) + "_" + str(split)).expect_partial()

                    trial_path = self.get_trial_dir(trial.trial_id)

                    #Save history output as a CSV
                    trainhist = pd.DataFrame(history.history)
                    trainhist.to_csv(self.__get_filename_path(trial_path, "history", ".csv", execution, split))

                    # Evaluate train performance on best epoch
                    obj_value = model.evaluate(traingen,
                                               return_dict=True,
                                               verbose=self._verbose,)

                    # Evaluate validation performance on best epoch
                    val_res = model.evaluate(valgen,
                                             return_dict=True,
                                             verbose=self._verbose,)

                    obj_value.update({"val_" + str(key): val for key, val in val_res.items()})

                    test_res = model.evaluate(testgen,
                                             return_dict=True,
                                             verbose=self._verbose,)

                    obj_value.update({"test_" + str(key): val for key, val in test_res.items()})

                    # Create and save CONFUSION MATRIX if requested
                    if self._model in ('DenseNet201', 'CNN'):
                        y_pred = model.predict(testgen, verbose=0)
                    elif self._model in ('DANNseNet201', 'DANN'):
                        y_pred, dom_pred = model.predict(testgen, verbose=0)
                    elif self._model in ('catDANN', 'CatDANN'):
                        y_pred, dom_pred = model.predict(testgen, verbose=0)

                    predicted_y = np.append(predicted_y, np.argmax(y_pred, axis=1))
                    true_y = np.append(true_y, np.argmax(y_test, axis=1))

                    if self._save_cm:
                        tot_predicted_y = np.append(tot_predicted_y, np.argmax(y_pred, axis=1))
                        tot_true_y = np.append(tot_true_y, np.argmax(y_test, axis=1)) 
                        results_df = generate_df(test_paths, true_y, predicted_y, classes, DANN=False)
                        results_df.to_csv(self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "_Val_predictions"), ".csv", execution, split))
                        report = classification_report(results_df['true_animal'], results_df['pred_animal'], 
                                                       labels=classes, target_names=classes, output_dict=True)
                        report_df=pd.DataFrame(report)
                        report_df['Fold'] = split
                        report_df.to_csv(self.__get_filename_path(trial_path, (self._model + '_' + self._group_name + '_Fold_metrics'), ".csv", execution, split))
                        self.__save_cm(true_y, predicted_y, class_names=classes, 
#                            filenameL=self.__get_filename_path(trial_path, "Confusion_Matrix", "light.png", execution, split),
                            filenameD=self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "_Confusion_Matrix"), "dark.png", execution, split))

                        if self._model in ('catDANN', 'CatDANN'):
                            true_dom = np.append(true_dom, np.argmax(domain_val, axis=1))
                            predicted_dom = np.append(predicted_dom, np.argmax(dom_pred, axis=1))

                            tot_predicted_dom = np.append(tot_predicted_dom, np.argmax(dom_pred, axis=1))
                            tot_true_dom = np.append(tot_true_dom, np.argmax(domain_val, axis=1))

                            resultsdf = generate_df(test_paths, true_y, predicted_y, classes, DANN=True, model=self._model, dom_true=true_dom, dom_pred=predicted_dom)
                            resultsdf.to_csv(self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "_Val_predictions"), ".csv", execution, split))
                            self.__save_cm(true_dom, predicted_dom, class_names=self._domain_names, 
                                filenameD=self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "_Season_Confusion_Matrix"), "dark.png", execution, split))

                    # Create and save DOMAIN PREDICTION VECTOR if requested
                    if self._save_domvec:
                        tot_predicted_dom = np.append(tot_predicted_dom, dom_pred)
                        tot_true_dom = np.append(tot_true_dom, sincos_val)
                        resultsdf = generate_df(test_paths, true_y, predicted_y, classes, DANN=True, model= self._model, dom_true=sincos_val, dom_pred=dom_pred)
                        resultsdf.to_csv(self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "_Val_predictions"), ".csv", execution, split))

                        self.__save_domvec(resultsdf, 
                                           Dom_filename=self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "_Domain_TruePred"), ".png", execution, split),
                                           DoY_filename=self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "_DoY_TruePred_LegendIn"), ".png", execution, split))

                     # Append training and validation scores to the histories
                    histories.append(obj_value)
                    reports.append(report)
            if self._save_cm:
                self.__save_cm(tot_true_y, tot_predicted_y, class_names=classes, 
                    filenameD=self.__get_filename_path(trial_path, "Averaged Confusion_Matrix", ".png", "final", "split"))
                if self._model in ('catDANN', 'CatDANN'):
                    self.__save_cm(tot_true_dom, tot_predicted_dom, class_names=self._domain_names, 
                        filenameD=self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "Averaged_Season_Confusion_Matrix"), "dark.png", execution, split))

            # It will returns an array of dictionary, note by default keras-tuner
            # will compute an average. This average is therefore the average of the
            # scores across the folds.
            reportsdf = pd.DataFrame(reports)
            reportsdf.to_csv(self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "_metrics_report"), ".csv", execution, split))

            histdf = pd.DataFrame(histories)
            histdf.to_csv(self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "_trial_scores"), ".csv", execution, split))
            return histories

        def __get_filename_path(self, trial_path, name, ext, execution, split):
            return os.path.join(trial_path, name + "_" + str(execution) + "_" + str(split) + ext,)

        def get_history(self, trial):
            histories = []
            trial_path = self.get_trial_dir(trial.trial_id)
            for split in range(self._kfoldcv.get_n_splits()):
                executions = []
                for execution in range(self.executions_per_trial):
                    with open(self.__get_filename_path(trial_path, "history", ".json", execution, split)) as fp:
                        executions.append(json.load(fp))
                histories.append(executions if len(executions) > 1 else executions[0])
            return histories

        def get_output(self, trial):
            outputs = []
            trial_path = self.get_trial_dir(trial.trial_id)
            for split in range(self._kfoldcv.get_n_splits()):
                executions = []
                for execution in range(self.executions_per_trial):
                    training = np.load(self.__get_filename_path(trial_path, "training", ".npy", execution, split))
                    validation = np.load(self.__get_filename_path(trial_path, "validation", ".npy", execution, split),)
                    executions.append((training, validation))
                outputs.append(executions if len(executions) > 1 else executions[0])
            return outputs

        def _build_and_fit_model(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            model = self._try_build(hp)
            history = self.hypermodel.fit(hp, model, *args, **kwargs)
            return history, model

        def __save_output(self, model, x, filename):
            y = model.predict(x,
                              batch_size=len(x),
                              verbose=self._display.verbose,)
            with open(filename, "wb",) as fp:
                np.save(fp, y)

        def __save_history(self, history, filename):
            with open(filename, "w",) as fp:
                json.dump(history.history, fp)

        def __save_cm(self, y_true, y_pred, class_names, filenameD):
            cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
            figureD = generate_confusion_matrix(cm, class_names, colour="dark")
            figureD.savefig(filenameD, dpi=300, bbox_inches='tight')

        def __save_domvec(self, df, Dom_filename, DoY_filename):
            Dom_figure = generate_domain_truepred(df)
            Dom_figure.savefig(Dom_filename, dpi=300)
            plt.close(Dom_figure)
            DoY_figure = generate_doy_truepred(df)
            DoY_figure.savefig(DoY_filename, dpi=300)
            plt.close(DoY_figure)

        def load_model(self, trial):
            """
            Returns all models associated with a specific trial. The output is an array where
            the number is determined by the number of splits of the cross validation. Each
            element of the array can be a single model if self.executions_per_trial is equal
            to 1, an array if it is greater.
            """
            models = []
            for split in range(self._kfoldcv.get_n_splits()):
                executions = []
                for execution in range(self.executions_per_trial):
                    model = self._try_build(trial.hyperparameters)
                    # Reload best checkpoint.
                    # Only load weights to avoid loading `custom_objects`.
                    with maybe_distribute(self.distribution_strategy):
                        model.load_weights(self._get_checkpoint_fname(trial.trial_id)
                            + "_" + str(execution)
                            + "_" + str(split))
                    executions.append(model)
                models.append(executions if len(executions) > 1 else executions[0])
            return models

    return SKF_CV

#%%
def Nested_DANNSKF_Tuner(superclass: Tuner,):
    class DANNSKF_CV(superclass):
        """
        Hyparameters search evaluated using Stratified Cross-Validation over a
        parameter space.
        """

        def __init__(self,
                     hypermodel,
                     kfoldcv: StratifiedKFold,
                     group: None,
                     x_train,
                     y_train,
                     path,
                     class_names,
#                     x_test: None,
 #                    y_test: None,
                     *args,
                     save_history=False,
                     save_output=False,
                     save_domvec=True,
                     save_cm=True,
                     restore_best=True,
                     batch_size,
                     **kwargs,):
            """Stratified CV Tuner constructor.

            Args:
                cv (BaseCrossValidator): instance of cross validator to use.
            """
            super(DANNSKF_CV, self).__init__(hypermodel, *args, **kwargs)
            self._kfoldcv = kfoldcv
            self._group = group
#            self._x_test=x_test
 #           self._y_test=y_test
            self._save_history = save_history
            self._save_output = save_output
            self._save_cm = save_cm
            self._save_domvec = save_domvec
            self._restore_best = restore_best
            self._batch_size = batch_size
            self._verbose = 2

            if self._kfoldcv == StratifiedGroupKFold and self._group is None:
                raise ValueError("Grouping argument must be included if using the StratifiedGroupKFold CV Tuner.")
                
        def search(self, *fit_args, **fit_kwargs):
            if "verbose" in fit_kwargs:
                self._verbose = fit_kwargs.get("verbose")
               # self._display._verbose = self._verbose
            self.on_search_begin()
            while True:
                trial = self.oracle.create_trial(self.tuner_id)
                if trial.status == trial_module.TrialStatus.STOPPED:
                    # Oracle triggered exit.
                    tf.get_logger().info("Oracle triggered exit")
                    break
                if trial.status == trial_module.TrialStatus.IDLE:
                    # Oracle is calculating, resend request.
                    continue

                self.on_trial_begin(trial)
                results = self.run_trial(trial, *fit_args, **fit_kwargs)
                # `results` is None indicates user updated oracle in `run_trial()`.
                if results is None:
                    warnings.warn("`Tuner.run_trial()` returned None. It should return one of "
                                  "float, dict, keras.callbacks.History, or a list of one "
                                  "of these types. The use case of calling "
                                  "`Tuner.oracle.update_trial()` in `Tuner.run_trial()` is "
                                  "deprecated, and will be removed in the future.",
                                  DeprecationWarning, stacklevel=2,)
                else:
                    metrics = tuner_utils.convert_to_metrics_dict(results, self.oracle.objective)
                    metrics.update(get_metrics_std_dict(results))
                    self.oracle.update_trial(trial.trial_id, metrics,)
                self.on_trial_end(trial)
            self.on_search_end()

        def run_trial(self, trial, x_train, y_train, path, class_names, *args, **kwargs):
            
            original_callbacks = kwargs.pop("callbacks", [])
            classes = class_names
            X = x_train
            if isinstance(y_train, dict):
                Y = y_train['out_class']
                sincoslabels = y_train['out_domain']
                tot_predicted_dom = np.array([])
                tot_true_dom = np.array([])
                DANN=True
            else: 
                Y = y_train
                DANN=False

            y_labels = tf.keras.utils.to_categorical(Y, num_classes=6)
            # Run the training process multiple times.
            histories = []
            tot_image_paths = np.array([])
            tot_predicted_y = np.array([])
            tot_true_y = np.array([])
            for execution in range(self.executions_per_trial):
                # Run the training over different splits.
                for split, (train_index, val_index) in enumerate(self._kfoldcv.split(X, Y, self._group)):
                    if self._verbose ==2: print("\n" + "-" * 30 + "\n"
                                                           f"K-fold Cross-Validation {split + 1}/{self._kfoldcv.get_n_splits()}"
                                                           + "\n" + "-" * 30 + "\n")
                    predicted_y = np.array([])
                    true_y = np.array([])
                    # Create a copy of args and kwargs to fill with fold-specific data
                    copied_args = []
                    copied_kwargs = copy.copy(kwargs)

                    if DANN==False:
                        # Get training set
                        x_train = X[train_index]
                        y_train = y_labels[train_index]
                        train_paths = path[train_index]
                        # Set the training set
                    
                        copied_kwargs["x"] = x_train
                        copied_kwargs["y"] = y_train
                        # Get the validation set
                        x_val = X[val_index]
                        y_val = y_labels[val_index]
                        val_paths = path[val_index]
                        # Set the validation set
                        copied_kwargs["validation_data"] = [x_val, y_val]
                    else:
                        # Get training set
                        x_train = X[train_index]
                        y_train = y_labels[train_index]
                        sincos_train = sincoslabels[train_index]
                        # Set the training set
                    
                        copied_kwargs["x"] = x_train
                        copied_kwargs["y"] = [y_train, sincos_train]
                        # Get the validation set
                        x_val = X[val_index]
                        val_paths = path[val_index]
                        y_val = y_labels[val_index]
                        sincos_val = sincoslabels[val_index]
                        # Set the validation set
                        copied_kwargs["validation_data"] = [x_val, [y_val, sincos_val]]
                    # -------------------------------------------------------
                    # Callbacks
                    # -------------------------------------------------------

                    # Configure tensorboard
                    callbacks = self._deepcopy_callbacks(original_callbacks)
                    self._configure_tensorboard_dir(callbacks, trial, str(execution) + "_" + str(split))
                    model_checkpoint = tuner_utils.SaveBestEpoch(objective=self.oracle.objective,
                                                                 filepath=self._get_checkpoint_fname(trial.trial_id) + "_" + str(execution) + "_" + str(split),)
                    
                    callbacks.append(tuner_utils.TunerCallback(self, trial))
                    # Save all the checkpoints.
                    # The file name will be checkpoint_{execution}_{split}
                    callbacks.append(model_checkpoint)
                    copied_kwargs["callbacks"] = callbacks

                    # Build and train the model
                    history, model = self._build_and_fit_model(trial, *copied_args, **copied_kwargs)

                    if self._restore_best:
                        # Load the best epoch according to objective function
                        model = self._try_build(trial.hyperparameters)
                        model.load_weights(self._get_checkpoint_fname(trial.trial_id)
                            + "_" + str(execution) + "_" + str(split)).expect_partial()

                    trial_path = self.get_trial_dir(trial.trial_id)

                    #Save history output as a CSV
                    trainhist = pd.DataFrame(history.history)
                    trainhist.to_csv(self.__get_filename_path(trial_path, "history", ".csv", execution, split))


                    # Save the history if requested
                    if self._save_history:
                        self.__save_history(history,
                            self.__get_filename_path(trial_path, "history", ".json", execution, split),)
                    # Save the output in numpy format if requested
                    if self._save_output:
                        self.__save_output(model, x_train,
                            self.__get_filename_path(trial_path, "training", ".npy", execution, split),)
                        self.__save_output(model, x_val,
                            self.__get_filename_path(trial_path, "validation", ".npy", execution, split),)

                    # Evaluate train performance on best epoch
                    obj_value = model.evaluate(x_train,
                                               {'out_class': y_train, 'out_domain': sincos_train}, #or just y_train
                                               batch_size=self._batch_size,
                                               return_dict=True,
                                               verbose=self._verbose,)

                    # Evaluate validation performance on best epoch
                    val_res = model.evaluate(x_val,
                                             {'out_class': y_val, 'out_domain': sincos_val}, #or just y_val
                                             batch_size=self._batch_size,
                                             return_dict=True,
                                             verbose=self._verbose,)
                    
                    obj_value.update({"val_" + str(key): val for key, val in val_res.items()})

#                    test_res = model.evaluate(Xt,
 #                                             yt_labels,
  #                                            batch_size=self._batch_size,
   #                                           return_dict=True,
    #                                          verbose=self._display.verbose,)
                    
       #             obj_value.update({"test_" + str(key): val for key, val in test_res.items()})

                    # Create and save CONFUSION MATRIX if requested
                    y_pred, dom_pred = model.predict(x_val)
                    tot_image_paths = np.append(tot_image_paths, val_paths)
                    predicted_y = np.append(predicted_y, np.argmax(y_pred, axis=1))
                    true_y = np.append(true_y, np.argmax(y_val, axis=1))
                    
                    if self._save_cm:
                        tot_predicted_y = np.append(tot_predicted_y, np.argmax(y_pred, axis=1))
                        tot_true_y = np.append(tot_true_y, np.argmax(y_val, axis=1)) 

                        self.__save_cm(true_y, predicted_y, class_names=classes, 
                            filenameL=self.__get_filename_path(trial_path, "Confusion_Matrix", "light.png", execution, split),
                            filenameD=self.__get_filename_path(trial_path, "Confusion_Matrix", "dark.png", execution, split))


                    # Create and save DOMAIN PREDICTION VECTOR if requested
                    if self._save_domvec:
                        tot_predicted_dom = np.append(tot_predicted_dom, dom_pred)
                        tot_true_dom = np.append(tot_true_dom, sincos_val)
                        df = generate_df(val_paths, true_y, predicted_y, DANN=True, dom_true=sincos_val, dom_pred=dom_pred)
                        df.to_csv(self.__get_filename_path(trial_path, "Val_predictions", ".csv", execution, split))

                        self.__save_domvec(df, 
                                           Dom_filename=self.__get_filename_path(trial_path, "Domain_TruePred", ".png", execution, split),
                                           DoY_filename_out=self.__get_filename_path(trial_path, "DoY_TruePred_legendOut", ".png", execution, split),
                                           DoY_filename_in=self.__get_filename_path(trial_path, "DoY_TruePred_LegendIn", ".png", execution, split))

                    # Append training and validation scores to the histories
                    histories.append(obj_value)
                    
            if self._save_cm:
                self.__save_cm(tot_true_y, tot_predicted_y, class_names=classes, 
                    filenameL=self.__get_filename_path(trial_path, "Averaged Confusion_Matrix", "light.png", "final", "split"),
                    filenameD=self.__get_filename_path(trial_path, "Averaged Confusion_Matrix", "dark.png", "final", "split"))
#            if self._save_domvec:
 #               df = generate_df(tot_image_paths, tot_true_y, tot_predicted_y, tot_true_dom, tot_predicted_dom)
  #              df.to_csv(self.__get_filename_path(trial_path, "Total_Val_predictions", ".csv", execution, split))

   #             self.__save_domvec(df, 
    #                               Dom_filename=self.__get_filename_path(trial_path, "Total_Domain_TruePred", ".png", execution, split),
     #                              DoY_filename_out=self.__get_filename_path(trial_path, "Total_DoY_TruePred_legendOut", ".png", execution, split),
      #                             DoY_filename_in=self.__get_filename_path(trial_path, "Total_DoY_TruePred_LegendIn", ".png", execution, split))


            # It will returns an array of dictionary, note by default keras-tuner
            # will compute an average. This average is therefore the average of the
            # scores across the folds.
            histdf = pd.DataFrame(histories)
            histdf.to_csv(self.__get_filename_path(trial_path, "DANNseNet201_Trial_Metric_Summaries", ".csv", execution, split))
            return histories

        def __get_filename_path(self, trial_path, name, ext, execution, split):
            return os.path.join(trial_path, name + "_" + str(execution) + "_" + str(split) + ext,)

        def get_history(self, trial):
            histories = []
            trial_path = self.get_trial_dir(trial.trial_id)
            for split in range(self._kfoldcv.get_n_splits()):
                executions = []
                for execution in range(self.executions_per_trial):
                    with open(self.__get_filename_path(trial_path, "history", ".json", execution, split)) as fp:
                        executions.append(json.load(fp))
                histories.append(executions if len(executions) > 1 else executions[0])
            return histories

        def get_output(self, trial):
            outputs = []
            trial_path = self.get_trial_dir(trial.trial_id)
            for split in range(self._kfoldcv.get_n_splits()):
                executions = []
                for execution in range(self.executions_per_trial):
                    training = np.load(self.__get_filename_path(trial_path, "training", ".npy", execution, split))
                    validation = np.load(self.__get_filename_path(trial_path, "validation", ".npy", execution, split),)
                    executions.append((training, validation))
                outputs.append(executions if len(executions) > 1 else executions[0])
            return outputs

        def _build_and_fit_model(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            model = self._try_build(hp)
            history = self.hypermodel.fit(hp, model, *args, **kwargs)

            return history, model

        def __save_output(self, model, x, filename):
            y = model.predict(x,
                              batch_size=len(x),
                              verbose=self._display.verbose,)
            with open(filename, "wb",) as fp:
                np.save(fp, y)

        def __save_history(self, history, filename):
            with open(filename, "w",) as fp:
                json.dump(history.history, fp)
        
        def __save_cm(self, y_true, y_pred, class_names, filenameL, filenameD):
            cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
            figureL = generate_confusion_matrix(cm, class_names, colour="light")
            figureL.savefig(filenameL, dpi=300, bbox_inches='tight')
            plt.close(figureL)
            figureD = generate_confusion_matrix(cm, class_names, colour="dark")
            figureD.savefig(filenameD, dpi=300, bbox_inches='tight')
            plt.close(figureD)
            
        def __save_domvec(self, df, Dom_filename, DoY_filename_in):
            Dom_figure = generate_domain_truepred(df)
            Dom_figure.savefig(Dom_filename, dpi=300)
            plt.close(Dom_figure)
            DoY_figure = generate_doy_truepred(df)
            DoY_figure.savefig(DoY_filename_in, dpi=300)
            plt.close(DoY_figure)

        def load_model(self, trial):
            """
            Returns all models associated with a specific trial. The output is an array where
            the number is determined by the number of splits of the cross validation. Each
            element of the array can be a single model if self.executions_per_trial is equal
            to 1, an array if it is greater.
            """
            models = []
            for split in range(self._kfoldcv.get_n_splits()):
                executions = []
                for execution in range(self.executions_per_trial):
                    model = self._try_build(trial.hyperparameters)
                    # Reload best checkpoint.
                    # Only load weights to avoid loading `custom_objects`.
                    with maybe_distribute(self.distribution_strategy):
                        model.load_weights(self._get_checkpoint_fname(trial.trial_id)
                            + "_" + str(execution)
                            + "_" + str(split))
                    executions.append(model)
                models.append(executions if len(executions) > 1 else executions[0])
            return models

    return DANNSKF_CV

#%%
def Nested_SinCosSKF_Tuner(superclass: Tuner,):
    class SinCosSKF_CV(superclass):
        """
        Hyparameters search evaluated using Stratified Cross-Validation over a
        parameter space.
        """

        def __init__(self,
                     hypermodel,
                     kfoldcv: StratifiedKFold,
                     group: None,
                     x_train,
                     y_train,
                     sin_train,
                     cos_train,
                     path,
                     class_names,
#                     x_test: None,
 #                    y_test: None,
                     *args,
                     save_history=False,
                     save_output=False,
                     save_domvec=True,
                     save_cm=True,
                     restore_best=True,
                     batch_size,
                     **kwargs,):
            """Stratified CV Tuner constructor.

            Args:
                cv (BaseCrossValidator): instance of cross validator to use.
            """
            super(SinCosSKF_CV, self).__init__(hypermodel, *args, **kwargs)
            self._kfoldcv = kfoldcv
            self._group = group
#            self._x_test=x_test
 #           self._y_test=y_test
            self._save_history = save_history
            self._save_output = save_output
            self._save_cm = save_cm
            self._save_domvec = save_domvec
            self._restore_best = restore_best
            self._batch_size = batch_size
            self._verbose = 2

            if self._kfoldcv == StratifiedGroupKFold and self._group is None:
                raise ValueError("Grouping argument must be included if using the StratifiedGroupKFold CV Tuner.")
                
        def search(self, *fit_args, **fit_kwargs):
            if "verbose" in fit_kwargs:
                self._verbose = fit_kwargs.get("verbose")
                self._display._verbose = self._verbose
            self.on_search_begin()
            while True:
                trial = self.oracle.create_trial(self.tuner_id)
                if trial.status == trial_module.TrialStatus.STOPPED:
                    # Oracle triggered exit.
                    tf.get_logger().info("Oracle triggered exit")
                    break
                if trial.status == trial_module.TrialStatus.IDLE:
                    # Oracle is calculating, resend request.
                    continue

                self.on_trial_begin(trial)
                results = self.run_trial(trial, *fit_args, **fit_kwargs)
                # `results` is None indicates user updated oracle in `run_trial()`.
                if results is None:
                    warnings.warn("`Tuner.run_trial()` returned None. It should return one of "
                                  "float, dict, keras.callbacks.History, or a list of one "
                                  "of these types. The use case of calling "
                                  "`Tuner.oracle.update_trial()` in `Tuner.run_trial()` is "
                                  "deprecated, and will be removed in the future.",
                                  DeprecationWarning, stacklevel=2,)
                else:
                    metrics = tuner_utils.convert_to_metrics_dict(results, self.oracle.objective)
                    metrics.update(get_metrics_std_dict(results))
                    self.oracle.update_trial(trial.trial_id, metrics,)
                self.on_trial_end(trial)
            self.on_search_end()

        def run_trial(self, trial, x_train, y_train, sin_train, cos_train, path, class_names, *args, **kwargs):
            
            original_callbacks = kwargs.pop("callbacks", [])
            classes = class_names
            X = x_train
            Y = y_train
            sinlabels = sin_train
            coslabels = cos_train
            tot_predicted_dom = np.array([])
            tot_true_dom = np.array([])

            y_labels = tf.keras.utils.to_categorical(Y, num_classes=6)
            # Run the training process multiple times.
            histories = []
            tot_image_paths = np.array([])
            tot_predicted_y = np.array([])
            tot_true_y = np.array([])
            for execution in range(self.executions_per_trial):
                # Run the training over different splits.
                for split, (train_index, val_index) in enumerate(self._kfoldcv.split(X, Y, self._group)):
                    if self._verbose ==2: tf.get_logger().info("\n" + "-" * 30 + "\n"
                                                           f"K-fold Cross-Validation {split + 1}/{self._kfoldcv.get_n_splits()}"
                                                           + "\n" + "-" * 30 + "\n")
                    predicted_y = np.array([])
                    true_y = np.array([])
                    predicted_sin = np.array([])
                    true_sin = np.array([])
                    predicted_cos = np.array([])
                    true_cos = np.array([])
                    # Create a copy of args and kwargs to fill with fold-specific data
                    copied_args = []
                    copied_kwargs = copy.copy(kwargs)

                    # Get training set
                    x_train = X[train_index]
                    y_train = y_labels[train_index]
                    sin_train = sinlabels[train_index]
                    cos_train = coslabels[train_index]
                    # Set the training set
                
                    copied_kwargs["x"] = x_train
                    copied_kwargs["y"] = [y_train, sin_train, cos_train]
                    # Get the validation set
                    x_val = X[val_index]
                    val_paths = path[val_index]
                    y_val = y_labels[val_index]
                    sin_val = sinlabels[val_index]
                    cos_val = coslabels[val_index]
                    # Set the validation set
                    copied_kwargs["validation_data"] = [x_val, [y_val, sin_val, cos_val]]
                    # -------------------------------------------------------
                    # Callbacks
                    # -------------------------------------------------------

                    # Configure tensorboard
                    callbacks = self._deepcopy_callbacks(original_callbacks)
                    self._configure_tensorboard_dir(callbacks, trial, str(execution) + "_" + str(split))
                    model_checkpoint = tuner_utils.SaveBestEpoch(objective=self.oracle.objective,
                                                                 filepath=self._get_checkpoint_fname(trial.trial_id) + "_" + str(execution) + "_" + str(split),)
                    
                    callbacks.append(tuner_utils.TunerCallback(self, trial))
                    # Save all the checkpoints.
                    # The file name will be checkpoint_{execution}_{split}
                    callbacks.append(model_checkpoint)
                    copied_kwargs["callbacks"] = callbacks

                    # Build and train the model
                    history, model = self._build_and_fit_model(trial, *copied_args, **copied_kwargs)

                    if self._restore_best:
                        # Load the best epoch according to objective function
                        model = self._try_build(trial.hyperparameters)
                        model.load_weights(self._get_checkpoint_fname(trial.trial_id)
                            + "_" + str(execution) + "_" + str(split)).expect_partial()

                    trial_path = self.get_trial_dir(trial.trial_id)

                    #Save history output as a CSV
                    trainhist = pd.DataFrame(history.history)
                    trainhist.to_csv(self.__get_filename_path(trial_path, "history", ".csv", execution, split))


                    # Save the history if requested
                    if self._save_history:
                        self.__save_history(history,
                            self.__get_filename_path(trial_path, "history", ".json", execution, split),)
                    # Save the output in numpy format if requested
                    if self._save_output:
                        self.__save_output(model, x_train,
                            self.__get_filename_path(trial_path, "training", ".npy", execution, split),)
                        self.__save_output(model, x_val,
                            self.__get_filename_path(trial_path, "validation", ".npy", execution, split),)

                    # Evaluate train performance on best epoch
                    obj_value = model.evaluate(x_train,
                                               {'out_class': y_train, 'sine': sin_train, 'cosine': cos_train}, #or just y_train
                                               batch_size=self._batch_size,
                                               return_dict=True,
                                               verbose=self._verbose,)

                    # Evaluate validation performance on best epoch
                    val_res = model.evaluate(x_val,
                                             {'out_class': y_val, 'sine': sin_val, 'cosine': cos_val}, #or just y_val
                                             batch_size=self._batch_size,
                                             return_dict=True,
                                             verbose=self._display.verbose,)
                    
                    obj_value.update({"val_" + str(key): val for key, val in val_res.items()})

#                    test_res = model.evaluate(Xt,
 #                                             yt_labels,
  #                                            batch_size=self._batch_size,
   #                                           return_dict=True,
    #                                          verbose=self._display.verbose,)
                    
       #             obj_value.update({"test_" + str(key): val for key, val in test_res.items()})

                    # Create and save CONFUSION MATRIX if requested
                    y_pred, sin_pred, cos_pred = model.predict(x_val)
                    predicted_sin = np.append(predicted_sin, sin_pred)
                    true_sin = np.append(true_sin, sin_val)
                    predicted_cos = np.append(predicted_cos, cos_pred)
                    true_cos = np.append(true_cos, cos_val)

                    dom_pred = np.asarray([y for y in zip(predicted_sin, predicted_cos)])
                    sincos_val = np.asarray([y for y in zip(sin_val, cos_val)])
                    tot_image_paths = np.append(tot_image_paths, val_paths)
                    predicted_y = np.append(predicted_y, np.argmax(y_pred, axis=1))
                    true_y = np.append(true_y, np.argmax(y_val, axis=1))
                    if self._save_cm:
                        tot_predicted_y = np.append(tot_predicted_y, np.argmax(y_pred, axis=1))
                        tot_true_y = np.append(tot_true_y, np.argmax(y_val, axis=1)) 

                        self.__save_cm(true_y, predicted_y, class_names=classes, 
                            filenameL=self.__get_filename_path(trial_path, "Confusion_Matrix", "light.png", execution, split),
                            filenameD=self.__get_filename_path(trial_path, "Confusion_Matrix", "dark.png", execution, split))

                    # Create and save DOMAIN PREDICTION VECTOR if requested
                    if self._save_domvec:
                        tot_predicted_dom = np.append(tot_predicted_dom, dom_pred)
                        tot_true_dom = np.append(tot_true_dom, sincos_val)
                        df = generate_df(val_paths, true_y, predicted_y, DANN=True, dom_true=sincos_val, dom_pred=dom_pred)
                        df.to_csv(self.__get_filename_path(trial_path, "Val_predictions", ".csv", execution, split))

                        self.__save_domvec(df, 
                                           Dom_filename=self.__get_filename_path(trial_path, "Domain_TruePred", ".png", execution, split),
                                           DoY_filename_out=self.__get_filename_path(trial_path, "DoY_TruePred_legendOut", ".png", execution, split),
                                           DoY_filename_in=self.__get_filename_path(trial_path, "DoY_TruePred_LegendIn", ".png", execution, split))

                    # Append training and validation scores to the histories
                    histories.append(obj_value)
                    
            if self._save_cm:
                self.__save_cm(tot_true_y, tot_predicted_y, class_names=classes, 
                    filenameL=self.__get_filename_path(trial_path, "Averaged Confusion_Matrix", "light.png", "final", "split"),
                    filenameD=self.__get_filename_path(trial_path, "Averaged Confusion_Matrix", "dark.png", "final", "split"))
            if self._save_domvec:
                df = generate_df(tot_image_paths, tot_true_y, tot_predicted_y, tot_true_dom, tot_predicted_dom)
                df.to_csv(self.__get_filename_path(trial_path, "Total_Val_predictions", ".csv", execution, split))

                self.__save_domvec(df, 
                                   Dom_filename=self.__get_filename_path(trial_path, "Total_Domain_TruePred", ".png", execution, split),
                                   DoY_filename_out=self.__get_filename_path(trial_path, "Total_DoY_TruePred_legendOut", ".png", execution, split),
                                   DoY_filename_in=self.__get_filename_path(trial_path, "Total_DoY_TruePred_LegendIn", ".png", execution, split))


            # It will returns an array of dictionary, note by default keras-tuner
            # will compute an average. This average is therefore the average of the
            # scores across the folds.
            return histories

        def __get_filename_path(self, trial_path, name, ext, execution, split):
            return os.path.join(trial_path, name + "_" + str(execution) + "_" + str(split) + ext,)

        def get_history(self, trial):
            histories = []
            trial_path = self.get_trial_dir(trial.trial_id)
            for split in range(self._kfoldcv.get_n_splits()):
                executions = []
                for execution in range(self.executions_per_trial):
                    with open(self.__get_filename_path(trial_path, "history", ".json", execution, split)) as fp:
                        executions.append(json.load(fp))
                histories.append(executions if len(executions) > 1 else executions[0])
            return histories

        def get_output(self, trial):
            outputs = []
            trial_path = self.get_trial_dir(trial.trial_id)
            for split in range(self._kfoldcv.get_n_splits()):
                executions = []
                for execution in range(self.executions_per_trial):
                    training = np.load(self.__get_filename_path(trial_path, "training", ".npy", execution, split))
                    validation = np.load(self.__get_filename_path(trial_path, "validation", ".npy", execution, split),)
                    executions.append((training, validation))
                outputs.append(executions if len(executions) > 1 else executions[0])
            return outputs

        def _build_and_fit_model(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            model = self._try_build(hp)
            history = self.hypermodel.fit(hp, model, *args, **kwargs)

            return history, model

        def __save_output(self, model, x, filename):
            y = model.predict(x,
                              batch_size=len(x),
                              verbose=self._display.verbose,)
            with open(filename, "wb",) as fp:
                np.save(fp, y)

        def __save_history(self, history, filename):
            with open(filename, "w",) as fp:
                json.dump(history.history, fp)
        
        def __save_cm(self, y_true, y_pred, class_names, filenameL, filenameD):
            cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
            figureL = generate_confusion_matrix(cm, class_names, colour="light")
            figureL.savefig(filenameL, dpi=300, bbox_inches='tight')
            plt.close(figureL)
            figureD = generate_confusion_matrix(cm, class_names, colour="dark")
            figureD.savefig(filenameD, dpi=300, bbox_inches='tight')
            plt.close(figureD)
            
        def __save_domvec(self, df, Dom_filename, DoY_filename_out, DoY_filename_in):
            Dom_figure = generate_domain_truepred(df)
            Dom_figure.savefig(Dom_filename, dpi=300)
            plt.close(Dom_figure)
            DoY_figure = generate_doy_truepred(df, "out")
            DoY_figure.savefig(DoY_filename_out, dpi=300)
            plt.close(DoY_figure)
            DoY_figure = generate_doy_truepred(df, "in")
            DoY_figure.savefig(DoY_filename_in, dpi=300)
            plt.close(DoY_figure)

        def load_model(self, trial):
            """
            Returns all models associated with a specific trial. The output is an array where
            the number is determined by the number of splits of the cross validation. Each
            element of the array can be a single model if self.executions_per_trial is equal
            to 1, an array if it is greater.
            """
            models = []
            for split in range(self._kfoldcv.get_n_splits()):
                executions = []
                for execution in range(self.executions_per_trial):
                    model = self._try_build(trial.hyperparameters)
                    # Reload best checkpoint.
                    # Only load weights to avoid loading `custom_objects`.
                    with maybe_distribute(self.distribution_strategy):
                        model.load_weights(self._get_checkpoint_fname(trial.trial_id)
                            + "_" + str(execution)
                            + "_" + str(split))
                    executions.append(model)
                models.append(executions if len(executions) > 1 else executions[0])
            return models

    return SinCosSKF_CV

#%%
def Nested_CatSKF_Tuner(superclass: Tuner,):
    class CatSKF_CV(superclass):
        """
        Hyparameters search evaluated using Stratified Cross-Validation over a
        parameter space.
        """

        def __init__(self,
                     hypermodel,
                     kfoldcv: StratifiedKFold,
                     group: None,
                     x_train,
                     y_train,
                     path,
                     class_names,
                     domain_names,
#                     x_test: None,
 #                    y_test: None,
                     *args,
                     save_history=False,
                     save_output=False,
                     save_domvec=True,
                     save_cm=True,
                     restore_best=True,
                     batch_size,
                     **kwargs,):
            """Stratified CV Tuner constructor.

            Args:
                cv (BaseCrossValidator): instance of cross validator to use.
            """
            super(CatSKF_CV, self).__init__(hypermodel, *args, **kwargs)
            self._kfoldcv = kfoldcv
            self._group = group
#            self._x_test=x_test
 #           self._y_test=y_test
            self._save_history = save_history
            self._save_output = save_output
            self._save_cm = save_cm
            self._save_domvec = save_domvec
            self._restore_best = restore_best
            self._batch_size = batch_size
            self._verbose = 2

            if self._kfoldcv == StratifiedGroupKFold and self._group is None:
                raise ValueError("Grouping argument must be included if using the StratifiedGroupKFold CV Tuner.")
                
        def search(self, *fit_args, **fit_kwargs):
            if "verbose" in fit_kwargs:
                self._verbose = fit_kwargs.get("verbose")
               # self._display._verbose = self._verbose
            self.on_search_begin()
            while True:
                trial = self.oracle.create_trial(self.tuner_id)
                if trial.status == trial_module.TrialStatus.STOPPED:
                    # Oracle triggered exit.
                    tf.get_logger().info("Oracle triggered exit")
                    break
                if trial.status == trial_module.TrialStatus.IDLE:
                    # Oracle is calculating, resend request.
                    continue

                self.on_trial_begin(trial)
                results = self.run_trial(trial, *fit_args, **fit_kwargs)
                # `results` is None indicates user updated oracle in `run_trial()`.
                if results is None:
                    warnings.warn("`Tuner.run_trial()` returned None. It should return one of "
                                  "float, dict, keras.callbacks.History, or a list of one "
                                  "of these types. The use case of calling "
                                  "`Tuner.oracle.update_trial()` in `Tuner.run_trial()` is "
                                  "deprecated, and will be removed in the future.",
                                  DeprecationWarning, stacklevel=2,)
                else:
                    metrics = tuner_utils.convert_to_metrics_dict(results, self.oracle.objective)
                    metrics.update(get_metrics_std_dict(results))
                    self.oracle.update_trial(trial.trial_id, metrics,)
                self.on_trial_end(trial)
            self.on_search_end()

        def run_trial(self, trial, x_train, y_train, path, class_names, domain_names, *args, **kwargs):
            
            original_callbacks = kwargs.pop("callbacks", [])
            classes = class_names
            seasons = domain_names
            X = x_train
            if isinstance(y_train, dict):
                Y = y_train['out_class']
                domainlabels = y_train['out_domain']
                dom_labels = tf.keras.utils.to_categorical(domainlabels, num_classes=4)
                predicted_dom = np.array([])
                true_dom = np.array([])
                tot_predicted_dom = np.array([])
                tot_true_dom = np.array([])
                DANN=True
            else: 
                Y = y_train
                DANN=False

            y_labels = tf.keras.utils.to_categorical(Y, num_classes=6)

            # Run the training process multiple times.
            histories = []
            tot_image_paths = np.array([])
            tot_predicted_y = np.array([])
            tot_true_y = np.array([])
            for execution in range(self.executions_per_trial):
                # Run the training over different splits.
                for split, (train_index, val_index) in enumerate(self._kfoldcv.split(X, Y, self._group)):
                    if self._verbose ==2: tf.get_logger().info("\n" + "-" * 30 + "\n"
                                                           f"K-fold Cross-Validation {split + 1}/{self._kfoldcv.get_n_splits()}"
                                                           + "\n" + "-" * 30 + "\n")
                    predicted_y = np.array([])
                    true_y = np.array([])
                    # Create a copy of args and kwargs to fill with fold-specific data
                    copied_args = []
                    copied_kwargs = copy.copy(kwargs)

                    if DANN==False:
                        # Get training set
                        x_train = X[train_index]
                        y_train = y_labels[train_index]
                        train_paths = path[train_index]
                        # Set the training set
                    
                        copied_kwargs["x"] = x_train
                        copied_kwargs["y"] = y_train
                        # Get the validation set
                        x_val = X[val_index]
                        y_val = y_labels[val_index]
                        val_paths = path[val_index]
                        # Set the validation set
                        copied_kwargs["validation_data"] = [x_val, y_val]
                    else:
                        # Get training set
                        x_train = X[train_index]
                        y_train = y_labels[train_index]
                        domain_train = dom_labels[train_index]
                        # Set the training set
                    
                        copied_kwargs["x"] = x_train
                        copied_kwargs["y"] = [y_train, domain_train]
                        # Get the validation set
                        x_val = X[val_index]
                        val_paths = path[val_index]
                        y_val = y_labels[val_index]
                        domain_val = dom_labels[val_index]
                        # Set the validation set
                        copied_kwargs["validation_data"] = [x_val, [y_val, domain_val]]
                    # -------------------------------------------------------
                    # Callbacks
                    # -------------------------------------------------------

                    # Configure tensorboard
                    callbacks = self._deepcopy_callbacks(original_callbacks)
                    self._configure_tensorboard_dir(callbacks, trial, str(execution) + "_" + str(split))
                    model_checkpoint = tuner_utils.SaveBestEpoch(objective=self.oracle.objective,
                                                                 filepath=self._get_checkpoint_fname(trial.trial_id) + "_" + str(execution) + "_" + str(split),)
                    
                    callbacks.append(tuner_utils.TunerCallback(self, trial))
                    # Save all the checkpoints.
                    # The file name will be checkpoint_{execution}_{split}
                    callbacks.append(model_checkpoint)
                    copied_kwargs["callbacks"] = callbacks

                    # Build and train the model
                    history, model = self._build_and_fit_model(trial, *copied_args, **copied_kwargs)

                    if self._restore_best:
                        # Load the best epoch according to objective function
                        model = self._try_build(trial.hyperparameters)
                        model.load_weights(self._get_checkpoint_fname(trial.trial_id)
                            + "_" + str(execution) + "_" + str(split)).expect_partial()

                    trial_path = self.get_trial_dir(trial.trial_id)

                    #Save history output as a CSV
                    trainhist = pd.DataFrame(history.history)
                    trainhist.to_csv(self.__get_filename_path(trial_path, "history", ".csv", execution, split))


                    # Save the history if requested
                    if self._save_history:
                        self.__save_history(history,
                            self.__get_filename_path(trial_path, "history", ".json", execution, split),)
                    # Save the output in numpy format if requested
                    if self._save_output:
                        self.__save_output(model, x_train,
                            self.__get_filename_path(trial_path, "training", ".npy", execution, split),)
                        self.__save_output(model, x_val,
                            self.__get_filename_path(trial_path, "validation", ".npy", execution, split),)

                    # Evaluate train performance on best epoch
                    obj_value = model.evaluate(x_train,
                                               {'out_class': y_train, 'out_domain': domain_train}, #or just y_train
                                               batch_size=self._batch_size,
                                               return_dict=True,
                                               verbose=self._verbose,)

                    # Evaluate validation performance on best epoch
                    val_res = model.evaluate(x_val,
                                             {'out_class': y_val, 'out_domain': domain_val}, #or just y_val
                                             batch_size=self._batch_size,
                                             return_dict=True,
                                             verbose=self._verbose,)
                    
                    obj_value.update({"val_" + str(key): val for key, val in val_res.items()})

#                    test_res = model.evaluate(Xt,
 #                                             yt_labels,
  #                                            batch_size=self._batch_size,
   #                                           return_dict=True,
    #                                          verbose=self._display.verbose,)
                    
       #             obj_value.update({"test_" + str(key): val for key, val in test_res.items()})

                    # Create and save CONFUSION MATRIX if requested
                    y_pred, dom_pred = model.predict(x_val)
                    tot_image_paths = np.append(tot_image_paths, val_paths)
                    predicted_y = np.append(predicted_y, np.argmax(y_pred, axis=1))
                    true_y = np.append(true_y, np.argmax(y_val, axis=1))
                    predicted_dom = np.append(predicted_dom, np.argmax(dom_pred, axis=1))
                    true_dom = np.append(true_dom, np.argmax(domain_val, axis=1))
                    
                    if self._save_cm:
                        tot_predicted_y = np.append(tot_predicted_y, np.argmax(y_pred, axis=1))
                        tot_true_y = np.append(tot_true_y, np.argmax(y_val, axis=1))
                        
                        tot_predicted_dom = np.append(tot_predicted_dom, np.argmax(dom_pred, axis=1))
                        tot_true_dom = np.append(tot_true_dom, np.argmax(domain_val, axis=1))

                        self.__save_cm(true_y, predicted_y, class_names=classes, 
                            filenameL=self.__get_filename_path(trial_path, "Confusion_Matrix", "light.png", execution, split),
                            filenameD=self.__get_filename_path(trial_path, "Confusion_Matrix", "dark.png", execution, split))
                        
                        self.__save_cm(true_dom, predicted_dom, class_names=seasons, 
                            filenameL=self.__get_filename_path(trial_path, "Season_Confusion_Matrix", "light.png", execution, split),
                            filenameD=self.__get_filename_path(trial_path, "Season_Confusion_Matrix", "dark.png", execution, split))

                    # Append training and validation scores to the histories
                    histories.append(obj_value)
                    
            if self._save_cm:
                self.__save_cm(tot_true_y, tot_predicted_y, class_names=classes, 
                    filenameL=self.__get_filename_path(trial_path, "Averaged Confusion_Matrix", "light.png", "final", "split"),
                    filenameD=self.__get_filename_path(trial_path, "Averaged Confusion_Matrix", "dark.png", "final", "split"))
                self.__save_cm(true_dom, predicted_dom, class_names=seasons, 
                    filenameL=self.__get_filename_path(trial_path, "Averaged Season_Confusion_Matrix", "light.png", execution, split),
                    filenameD=self.__get_filename_path(trial_path, "Averaged Season_Confusion_Matrix", "dark.png", execution, split))

            # It will returns an array of dictionary, note by default keras-tuner
            # will compute an average. This average is therefore the average of the
            # scores across the folds.
            histdf = pd.DataFrame(histories)
            histdf.to_csv(self.__get_filename_path(trial_path, "DANNseNet201_Trial_Metric_Summaries", ".csv", execution, split))
            return histories

        def __get_filename_path(self, trial_path, name, ext, execution, split):
            return os.path.join(trial_path, name + "_" + str(execution) + "_" + str(split) + ext,)

        def get_history(self, trial):
            histories = []
            trial_path = self.get_trial_dir(trial.trial_id)
            for split in range(self._kfoldcv.get_n_splits()):
                executions = []
                for execution in range(self.executions_per_trial):
                    with open(self.__get_filename_path(trial_path, "history", ".json", execution, split)) as fp:
                        executions.append(json.load(fp))
                histories.append(executions if len(executions) > 1 else executions[0])
            return histories

        def get_output(self, trial):
            outputs = []
            trial_path = self.get_trial_dir(trial.trial_id)
            for split in range(self._kfoldcv.get_n_splits()):
                executions = []
                for execution in range(self.executions_per_trial):
                    training = np.load(self.__get_filename_path(trial_path, "training", ".npy", execution, split))
                    validation = np.load(self.__get_filename_path(trial_path, "validation", ".npy", execution, split),)
                    executions.append((training, validation))
                outputs.append(executions if len(executions) > 1 else executions[0])
            return outputs

        def _build_and_fit_model(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            model = self._try_build(hp)
            history = self.hypermodel.fit(hp, model, *args, **kwargs)

            return history, model

        def __save_output(self, model, x, filename):
            y = model.predict(x,
                              batch_size=len(x),
                              verbose=self._display.verbose,)
            with open(filename, "wb",) as fp:
                np.save(fp, y)

        def __save_history(self, history, filename):
            with open(filename, "w",) as fp:
                json.dump(history.history, fp)
        
        def __save_cm(self, y_true, y_pred, class_names, filenameL, filenameD):
            cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
            figureL = generate_confusion_matrix(cm, class_names, colour="light")
            figureL.savefig(filenameL, dpi=300, bbox_inches='tight')
            plt.close(figureL)
            figureD = generate_confusion_matrix(cm, class_names, colour="dark")
            figureD.savefig(filenameD, dpi=300, bbox_inches='tight')
            plt.close(figureD)
            
        def __save_domvec(self, df, Dom_filename, DoY_filename_out, DoY_filename_in):
            Dom_figure = generate_domain_truepred(df)
            Dom_figure.savefig(Dom_filename, dpi=300)
            plt.close(Dom_figure)
            DoY_figure = generate_doy_truepred(df, "out")
            DoY_figure.savefig(DoY_filename_out, dpi=300)
            plt.close(DoY_figure)
            DoY_figure = generate_doy_truepred(df, "in")
            DoY_figure.savefig(DoY_filename_in, dpi=300)
            plt.close(DoY_figure)

        def load_model(self, trial):
            """
            Returns all models associated with a specific trial. The output is an array where
            the number is determined by the number of splits of the cross validation. Each
            element of the array can be a single model if self.executions_per_trial is equal
            to 1, an array if it is greater.
            """
            models = []
            for split in range(self._kfoldcv.get_n_splits()):
                executions = []
                for execution in range(self.executions_per_trial):
                    model = self._try_build(trial.hyperparameters)
                    # Reload best checkpoint.
                    # Only load weights to avoid loading `custom_objects`.
                    with maybe_distribute(self.distribution_strategy):
                        model.load_weights(self._get_checkpoint_fname(trial.trial_id)
                            + "_" + str(execution)
                            + "_" + str(split))
                    executions.append(model)
                models.append(executions if len(executions) > 1 else executions[0])
            return models

    return CatSKF_CV

#%%
class Outer_SKF_Tuner:
    def __init__(self,
                 kfoldcv: StratifiedKFold,
                 tuner_class,
                 group = None,
                 *args,
                 multiprocess=False,
                 **kwargs,):
        """OuterCV constructor.

        Args:
            cv (BaseCrossValidator): instance of cross validator to use.
        """
        if len(args) > 0:
            self._build_model = args[0]
        else:
            self._build_model = kwargs.get("hypermodel")
        self._kfoldcv = kfoldcv
        self._group = group
        self._tuners = []
        self._output_dirs = []
        for i in range(kfoldcv.get_n_splits()):
            copied_kwargs = copy.copy(kwargs)
            copied_kwargs["directory"] = os.path.join(kwargs["directory"], "outer_cv_" + str(i)
            )
            self._output_dirs.append(
                os.path.join(copied_kwargs["directory"], copied_kwargs["project_name"])
            )
            self._tuners.append(tuner_class(*args, **copied_kwargs))
        self._verbose = True
        self.random_state = None
        self._multiprocess = multiprocess

    def _execute_inner_search(self, tuner, x_train, y_train, args, kwargs):
        copied_args, copied_kwargs = self._compute_training_args(x_train, y_train, *args, **kwargs)

        # Hyperparameter optimization
        tuner.search(*copied_args, **copied_kwargs)

    def search(self, *args, **kwargs):
        if "verbose" in kwargs:
            self._verbose = kwargs.get("verbose")

        X = args[0]
        Y = args[1]
        y_labels = tf.keras.utils.to_categorical(Y, num_classes=6)
        for split, (train_index, test_index) in enumerate(self._kfoldcv.split(X, Y, self._group)):
            if self._verbose==2:
                tf.get_logger().info("\n" + "-" * 30 + "\n"
                                     f"[Search] Outer Cross-Validation {split + 1}/{self._kfoldcv.get_n_splits()}"
                                     + "\n" + "-" * 30 + "\n")

            # Training split and execute search
            if self._multiprocess and split != self._kfoldcv.get_n_splits() - 1:
                process = Process(target=self._execute_inner_search, args=(self._tuners[split],
                                                                           X[train_index],
                                                                           y_labels[train_index],
                                                                           args,
                                                                           kwargs,),)
                process.start()
            else:
                self._execute_inner_search(self._tuners[split], X[train_index], y_labels[train_index], args, kwargs)

    def evaluate(self, *args, restore_best=False, **kwargs):
        if "verbose" in kwargs:
            self._verbose = kwargs.get("verbose")

        X = args[0]
        Y = args[1]
        y_labels = tf.keras.utils.to_categorical(Y, num_classes=6)
        epochs = kwargs.get("epochs")

        results = []
        for split, (train_index, test_index) in enumerate(self._kfoldcv.split(X, Y, self._group)):
            if self._verbose==2:
                tf.get_logger().info("\n" + "-" * 30 + "\n"
                                     f"[Evaluate] Outer Cross-Validation {split + 1}/{self._kfoldcv.get_n_splits()}"
                                     + "\n" + "-" * 30 + "\n")

            # Training split
            x_train = X[train_index]
            y_train = y_labels[train_index]

            # Test split
            x_test = X[test_index]
            y_test = y_labels[test_index]

            # Re-fit best model found during search
            tuner = self._tuners[split]
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = self._build_model(best_hps)
            copied_args, copied_kwargs = self._compute_training_args(x_train, y_train, *args, **kwargs)
            if restore_best:
                model_path = os.path.join(self._output_dirs[split], "best_model")
                if not "callbacks" in copied_kwargs:
                    copied_kwargs["callbacks"] = []
                copied_kwargs["callbacks"].append(tuner_utils.SaveBestEpoch(objective=tuner.oracle.objective,
                                                                            filepath=model_path,))
            if isinstance(epochs, list):
                copied_kwargs["epochs"] = epochs[split]
            model.fit(*copied_args, **copied_kwargs)

            # Restore best weight according to validation score
            if restore_best:
                model = self._build_model(best_hps)
                model.load_weights(model_path).expect_partial()

            # Compute training score
            result = self._evaluate(model, copied_args[0], copied_args[1])
            # Compute validation score
            if "validation_data" in copied_kwargs:
                validation_data = copied_kwargs.get("validation_data")
                result.update(self._evaluate(model, validation_data[0], validation_data[1], "val_"))
            # Compute test score
            result.update(self._evaluate(model, x_test, y_test, "test_"))

            results.append(result)

        # Compute average score across outer folds
        result = tuner_utils.average_metrics_dicts(results)
        # Compute standard deviation across outer folds
        result.update(get_metrics_std_dict(results))

        return result

    def get_tuners(self):
        return self._tuners

    def get_best_hparams(self):
        results = []
        for i in range(self._kfoldcv.get_n_splits()):
            results.append(self._tuners[i].get_best_hyperparameters(num_trials=1)[0])
        return results

    def _evaluate(self, model, x, y, prefix=""):
        evaluation = model.evaluate(x, y, batch_size=len(x), return_dict=True)
        return {prefix + str(key): val for key, val in evaluation.items()}

    def _compute_training_args(self, x_train, y_train, *args, **kwargs):
        copied_kwargs = copy.copy(kwargs)

        if "validation_split" in kwargs:
            copied_kwargs.pop("validation_split")
            x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                              y_train,
                                                              test_size=kwargs.get("validation_split"),
                                                              random_state=self.random_state,)
            copied_kwargs["validation_data"] = (x_val, y_val)
            if ("validation_batch_size" in kwargs and kwargs.get("validation_batch_size") == "full-batch"):
                copied_kwargs["validation_batch_size"] = len(x_val)

        # Compute full-batch size for training data
        if "batch_size" in kwargs and kwargs.get("batch_size") == "full-batch":
            copied_kwargs["batch_size"] = len(x_train)

        copied_args = []
        for arg in args:
            copied_args.append(arg)
        copied_args[0] = x_train
        copied_args[1] = y_train
        copied_args = tuple(arg for arg in copied_args)

        return copied_args, copied_kwargs

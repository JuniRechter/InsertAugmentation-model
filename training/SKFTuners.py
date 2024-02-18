# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:22:36 2023

@author: Juniper Rechter
"""
import copy
import warnings
import os
import json

import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
import ImageDataGenerators as idg
from keras_tuner.engine.tuner import Tuner, maybe_distribute

from keras_tuner.src import utils
from keras_tuner.src.engine import tuner_utils
from keras_tuner.src.engine import trial as trial_module
from keras_tuner_cv.utils import get_metrics_std_dict

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import itertools
import sklearn.metrics
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
        
    thresh = 0.6
    
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
def generate_df(path, class_true, class_pred, classes, DANN=False, dom_true=None, dom_pred=None):
    
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
        else:
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
                     group: None,
                     df,
                     dataset,
                     model_name,
                     class_names,
                     domain_names:None,
                     group_name,
                     *args,
                     save_history=False,
                     save_output=False,
                     save_cm=True,
                     save_domvec=False,
                     restore_best=True,
                     batch_size,
                     **kwargs,):
            """Stratified CV Tuner constructor.

            Args:
                cv (BaseCrossValidator): instance of cross validator to use.
            """
            super(SKF_CV, self).__init__(hypermodel, *args, **kwargs)
            self._kfoldcv = kfoldcv
            self._group = group
            self.df = df.copy()
            self._dataset = dataset
            self._model = model_name
            self._domain_names = domain_names
            self._group_name = group_name
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
            if self._dataset == "AHC":
                df['id'] = df['id']-1
            Y = np.array(df['id'])
            num_classes = (df['id'].nunique())
#            if self._model in ("DANNseNet201"):
 #               df['adjsin_date2'] = df['adjsin_date']*0.8
  #              df['adjcos_date2'] = df['adjcos_date']*0.8

#    class_weights = dict(zip(np.unique(y_train),
 #                            class_weight.compute_class_weight(class_weight = 'balanced',
  #                                                             classes= np.unique(y_train),
   #                                                            y= y_train)))

            histories = []
            tot_predicted_y = np.array([])
            tot_true_y = np.array([])
            tot_predicted_dom = np.array([])
            tot_true_dom = np.array([])

            # Run the training process multiple times.            
            for execution in range(self.executions_per_trial):
                # Run the training over different splits.
                for split, (train_index, val_index) in enumerate(self._kfoldcv.split(df, Y, self._group)):
                    if self._verbose ==2: tf.get_logger().info("\n" + "-" * 30 + "\n"
                                                           f"K-fold Cross-Validation {split + 1}/{self._kfoldcv.get_n_splits()}"
                                                           + "\n" + "-" * 30 + "\n")

                    train_df = df.iloc[train_index]
                    val_df = df.iloc[val_index]

                    traingen = idg.TrainingDataGenerator(train_df,
                                                           batch_size = self._batch_size,
                                                           y_col = {'id': 'id', 'domain': 'domain', 
                                                                    'sin_date': 'adjsin_date', 'cos_date': 'adjcos_date'},
                                                           X_col = {'path': 'path'},
                                                           model_name=self._model,
                                                           shuffle = True)
                    valgen = idg.ValidationDataGenerator(val_df,
                                                         batch_size = self._batch_size,
                                                         y_col = {'id': 'id', 'domain': 'domain', 
                                                                  'sin_date': 'adjsin_date', 'cos_date': 'adjcos_date'},
                                                         X_col = {'path': 'path'},
                                                         model_name=self._model,
                                                         shuffle = True)
                    testgen = idg.ValidationDataGenerator(val_df,
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
                    val_paths = np.array(val_df["path"])
                    y_val = np.array(val_df['id'])
                    y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

                    if self._model in ("DANNseNet201"):
                        sin_date = np.array(val_df['adjsin_date'])
                        cos_date = np.array(val_df['adjcos_date'])
                        sincos_val = np.asarray([y for y in zip(sin_date, cos_date)])
                    elif self._model in ('CatDANN', 'catDANN'):
                        cat_domain = np.array(val_df['domain'])
                        true_dom = tf.keras.utils.to_categorical(cat_domain, num_classes=4)

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

                    # Create and save CONFUSION MATRIX if requested
                    if self._model in ('DenseNet201', 'CNN'):
                        y_pred = model.predict(testgen, verbose=0)
                    elif self._model in ('DANNseNet201', 'DANN'):
                        y_pred, dom_pred = model.predict(testgen, verbose=0)
                    elif self._model in ('catDANN', 'CatDANN'):
                        y_pred, predicted_dom = model.predict(testgen, verbose=0)

                    predicted_y = np.append(predicted_y, np.argmax(y_pred, axis=1))
                    true_y = np.append(true_y, np.argmax(y_val, axis=1))
                    if self._save_cm:
                        tot_predicted_y = np.append(tot_predicted_y, np.argmax(y_pred, axis=1))
                        tot_true_y = np.append(tot_true_y, np.argmax(y_val, axis=1)) 
                        results_df = generate_df(val_paths, true_y, predicted_y, classes, DANN=False)
                        results_df.to_csv(self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "_Val_predictions"), ".csv", execution, split))

                        self.__save_cm(true_y, predicted_y, class_names=classes, 
#                            filenameL=self.__get_filename_path(trial_path, "Confusion_Matrix", "light.png", execution, split),
                            filenameD=self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "_Confusion_Matrix"), "dark.png", execution, split))

                        if self._model in ('catDANN', 'CatDANN'):
                            tot_predicted_dom = np.append(tot_predicted_dom, dom_pred)
                            tot_true_dom = np.append(tot_true_dom, true_dom)
                            self.__save_cm(true_dom, predicted_dom, class_names=self._domain_names, 
                                filenameD=self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "_Season_Confusion_Matrix"), "dark.png", execution, split))

                    # Create and save DOMAIN PREDICTION VECTOR if requested
                    if self._save_domvec:
                        tot_predicted_dom = np.append(tot_predicted_dom, dom_pred)
                        tot_true_dom = np.append(tot_true_dom, sincos_val)
                        resultsdf = generate_df(val_paths, true_y, predicted_y, classes, DANN=True, dom_true=sincos_val, dom_pred=dom_pred)
                        resultsdf.to_csv(self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "_Val_predictions"), ".csv", execution, split))

                        self.__save_domvec(resultsdf, 
                                           Dom_filename=self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "_Domain_TruePred"), ".png", execution, split),
                                           DoY_filename=self.__get_filename_path(trial_path, (self._model +'_'+ self._group_name + "_DoY_TruePred_LegendIn"), ".png", execution, split))

                     # Append training and validation scores to the histories
                    histories.append(obj_value)

            if self._save_cm:
                self.__save_cm(tot_true_y, tot_predicted_y, class_names=classes, 
                    filenameD=self.__get_filename_path(trial_path, "Averaged Confusion_Matrix", ".png", "final", "split"))

            # It will returns an array of dictionary, note by default keras-tuner
            # will compute an average. This average is therefore the average of the
            # scores across the folds.
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

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:28:58 2023
@author: Juniper Rechter

This file contains functions for visualising the output of the DANNseNet201 
model and related DenseNet201 models. 

These include:
    - generating a Confusion Matrix of animal class predictions;
    - generating a Dataframe of Model Predictions, including original image 
    name and true labels;
    - generating a SinCos mapping of model domain predictions in comparison to 
    true labels; and
    - generating a Day of Year (DoY) mapping of model domain predictions in 
    comparison to true labels.

This code also includes the current palettes used by these functions, including 
continuous linearly interpolated light and dark palettes (Matplotlib Inferno) 
for classification confusion matrices, and a palette for categorising the SinCos
mappings and DoY to the four meteorological seasons for intepreting domain 
outputs by the DANNseNet201.

All of these palettes are colourblind-friendly.

"""

import numpy as np
import pandas as pd

import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

#%% Graphing Palettes
'''
Note: The 'light' palette here depicts more frequent predictions in gold;
whereas the 'dark' palette depicts more frequent predictions in purple.
These palettes are reverses of each other.
Ideally, the true positives are represented in 'light' or 'dark', if your model
is accurate.
Therefore, the 'dark' palette is friendlier for printing in greyscale, whereas
the 'light' palette will generally use more ink. 

Generally, I use the 'dark' palette.

'''

light_palette = ["#221330", "#390962", "#5F126E", 
           "#85206A", "#A92E5E", "#CB4049", 
           "#E65C2E", "#FBAE12", "#F4DB4B", "#FFF999"]

dark_palette =["#FFF999", "#F4DB4B", "#FBAE12", "#E65C2E",
               "#CB4049", "#A92E5E", "#85206A",
               "#5F126E", "#390962", "#221330"]

season_palette = {"Fall": '#E30513', "Winter": '#78D2EB', 
                  "Spring": '#62D658', "Summer": '#D68802'}

#%%
def generate_confusion_matrix(cnf_matrix, classes, colour='light'):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Confusion matrix will adjust to the number of classes supplied.

    Inputs:
    - cm (array, shape = [n, n]): a sklearn.metrics.confusion matrix of integer classes.
    
    - class_names (array, shape = [n]): String names of the integer classes.
    
    - colour: str, 'light' or 'dark'.
    Note: the light palette depicts more frequent predictions in gold, whilst 
    the 'dark' palette depicts more frequent predictions in purple.
    The 'dark' palette is more friendly for printing in grey-scale, at least, 
    if your model is playing nice..
    
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
def generate_df(path, class_true, class_pred, DANN=False, dom_true=None, dom_pred=None):
    """
    Returns a pandas dataframe containing true and predicted labels for 
    model.predicted images. 
    This function will also provide additional transformed
    variables for human interpretation. These include transforming class labels 
    to the species name, and domain labels to the predicted Day of Year and 
    meteorological season.
    
    Df currently applies species labels to six majority classes.

    Inputs:
    - path (array, shape = [n]): string names of image paths
        
    - class_true (array, shape = [n]): an array of true class labels
        
    - class_pred (array, shape = [n]): an array of predicted class labels
        
    - DANN: bool, set whether or not domain labels will be provided. 
    default = False.
        
    - dom_true: (array, shape = [n, n]): an array of true [sine, cosine] labels,
    default = None, must be supplied if DANN == True.
        
    - dom_pred: (array, shape = [n, n]): an array of predicted [sine, cosine] labels,
    default = None, must be supplied if DANN == True.
        
    Output: the resulting dataframe, including image paths and species/season labels.
    """
    df=pd.DataFrame()
    image_path=path.tolist()
    df['image_path']=image_path
    
    class_dict = {0:'moose', 1:'fox', 2:'deer',
                  3:'sandhill crane', 4:'bear', 5:'domestic dog'}
    
    true_class=class_true.tolist()
    df['true_class']=pd.DataFrame(true_class)
    df['true_animal'] = df['true_class'].apply(lambda x: class_dict[x])

    pred_class=class_pred.tolist()
    df['pred_class']=pd.DataFrame(pred_class)
    df['pred_animal'] = df['pred_class'].apply(lambda x: class_dict[x])
    
    if DANN==True:
        if dom_true and dom_pred is None:
            print("Error: dom_true or dom_pred were not provided for df." + \
                  "Returning df with class predictions only.")
            return df
        else:
            true_dom=dom_true.tolist()
            df[['true_sine', 'true_cosine']] = pd.DataFrame(true_dom)

            df['true_DoY']=((np.arctan2(df.true_sine, df.true_cosine))*365)/(2*np.pi)
            df['true_DoY'] = [(365 + ele) if ele <0 else ele for ele in df['true_DoY']]
    
            pred_dom=dom_pred.tolist()
            df[['pred_sine', 'pred_cosine']] = pd.DataFrame(pred_dom)
            df['pred_DoY']=((np.arctan2(df.pred_sine, df.pred_cosine))*365)/(2*np.pi)
            df['pred_DoY'] = [(365 + ele) if ele <0 else ele for ele in df['pred_DoY']]

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
    Returns a matplotlib figure containing the plotted domain labels and 
    predictions, with the x-axis representing Sine normalised values, and the 
    y-axis representing Cosine normalised values.
    
    Figure uses the 'dark' palette to depict True Labels as purple diamonds, 
    and Predicted Labels as gold circles. Lines are also produced to depict 
    corresponding true and predicted labels for an image.

    Inputs:
    - df: The dataframe generated from the above function, or one which includes
    variables 'pred_sine', 'pred_cosine', 'true_sine' and 'true_cosine' for each
    image evaluated.
    
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

# If monthly markers are desired, uncomment these points.
    #plt.annotate(text="January", xy=[0.0172, 1.05])
    #plt.annotate(text="February",xy=[0.5221, 0.9529])
    #plt.annotate(text="March",xy=[0.95, 0.5647])
    #plt.annotate(text="April", xy=[1.08, 0.0085])
    #plt.annotate(text="May",xy=[0.93, -0.53])
    #plt.annotate(text="June",xy=[0.5074, -0.9617])
    #plt.annotate(text="July",xy=[0.0172, -1.05])
    #plt.annotate(text="August",xy=[-0.4925, -0.9702])
    #plt.annotate(text="September",xy=[-1.01, -0.53])
    #plt.annotate(text="October",xy=[-1.125, -0.0085])
    #plt.annotate(text="November",xy=[-0.995, 0.5647])
    #plt.annotate(text="December",xy=[-0.5804, 0.9529])

    plt.annotate(text="Fall",xy=[-1.0, -0.52], fontsize=15)
    plt.annotate(text="Winter",xy=[-0.57, 0.92], fontsize=15)
    plt.annotate(text="Spring",xy=[0.91, 0.48], fontsize=15)
    plt.annotate(text="Summer",xy=[0.4, -1.0], fontsize=15)

    plt.ylabel('Cosine Normalised Time', labelpad=0.5, fontsize=17) #fontweight="bold")
    plt.xlabel('Sine Normalised Time', fontsize=17) #fontweight="bold")
    axis.legend(fontsize=15)
    
    return figure
#%%
def generate_doy_truepred(df, legend="in"):
    """
    Returns a matplotlib figure containing the plotted domain labels and predictions,
    with the x-axis representing True Day of Year (DoY) labels, and the y-axis 
    representing Predicted DoY labels, as produced within the df generator above.
    
    Figure uses the 'season' palette to True Winter DoYs as pale blue, True 
    Spring DoYs as green, True Summer DoYs as orange, and True Fall DoYs as red.
    These provide a marker for where the meteorological seasons in the DoY, 
    (ie. Winter - Dec 1, Spring - Mar 1, Summer - Jun 1, Fall - Sept 1), therefore
    showing if the model is overpredicting a particular season.

    Inputs:
    - df: The dataframe generated from the above function, or one which includes
    variables 'true_season', 'true_DoY', and 'pred_DoY'.
    
    - legend: str, set where the figure legend appears.
    Enter either "in" or "out". Default: "in"
    
    """
    
    figure = plt.figure(figsize=(12,12))
    axis = figure.add_subplot(111)
    cdict = season_palette 
    grouped = df.groupby("true_season")

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
    if legend=="out":
        axis.legend(handles=handles, bbox_to_anchor=(1.2, 0.593), title="True Season", title_fontsize=15, fontsize=15)
    else:
        axis.legend(handles=handles, title="True Season", title_fontsize=15, fontsize=15) #bbox_to_anchor=(1.2, 1.01), 

    return figure
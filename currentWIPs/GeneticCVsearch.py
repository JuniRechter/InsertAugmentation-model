# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:57:01 2023

@author: hayle
"""

import matplotlib.pyplot as plt
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous
from sklearn.model_selection import train_test_split, StratifiedKFold
from scikeras.wrappers import KerasClassifier

from sklearn.metrics import accuracy_score

#models
import tensorflow as tf
from tensorflow.keras.applications import densenet
import tensorflow.keras.layers as kl
from tensorflow.keras.optimizers import Adam

#%%
'''
https://github.com/gabriben/metrics-as-losses
https://openreview.net/pdf?id=gvSHaaD2wQ
'''

def weightedF1(y, y_hat):
    """Compute the weighted F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, 0.5), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1
#%%
#Construct DenseNet201 base model
def DenseNet201model(neurons, learn_rate):
    model = densenet.DenseNet201(weights="imagenet", include_top=False, input_shape=(224,224,3), pooling="avg")
    model.trainable = False

    out = kl.Dense(neurons, activation='relu')(model.layers[-1].output)
    out_class = kl.Dense(6, activation='softmax', name="out_class")(out)

    model = tf.keras.Model(
             inputs=model.inputs,
             outputs=out_class)

#    lr=hp.Float("lr", min_value=0.00016, max_value=0.0002, sampling="log")
    opt=Adam(learning_rate=learn_rate)
    metrics=["accuracy", 
             #weightedF1,
              tf.keras.metrics.Precision(name="precision"),
              tf.keras.metrics.Recall(name="recall")]

    model.compile(optimizer=opt,
                  loss="categorical_crossentropy",
                  metrics=metrics)
    return model

model = KerasClassifier(model=DenseNet201model, verbose=2)
#%%
param_grid= {
    "neurons": Integer(64, 512),
    "epochs": Integer(32, 64, 96),
    "learn_rate":Continuous(1e-4, 1e-1),
    "batch_size":Integer(32, 96)}
#%%
genetic_search = GASearchCV(estimator=model,
                            scoring="f1_weighted",
                            criteria="max",
                            param_grid=param_grid)
#%%
evolved_estimator = GASearchCV(estimator=model,
                               cv=cv,
                               scoring='val_accuracy',
                               population_size=10,
                               generations=35,
                               tournament_size=3,
                               elitism=True,
                               crossover_probability=0.8,
                               mutation_probability=0.1,
                               param_grid=param_grid,
                               criteria='max',
                               algorithm='eaMuPlusLambda',
                               n_jobs=-1,
                               verbose=True,
                               keep_top_k=4)
#%%
evolved_estimator.fit(traingen)

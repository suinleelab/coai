from collections import defaultdict
import itertools, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer as Imputer

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

DEFAULT_PARAMS = {'optimizer': 'rmsprop',
                  'loss':tf.keras.losses.BinaryCrossentropy(from_logits=False),
                 'layers': [128], 'pdrop': 0.0,'estimator_type':'classification'}

class ProperKerasClassifier(tf.keras.wrappers.scikit_learn.KerasClassifier):
    def __init__(self,*args,**kwargs):
        self._estimator_type='classifier'
        super().__init__(*args,**kwargs)
class ProperKerasRegressor(tf.keras.wrappers.scikit_learn.KerasClassifier):
    def __init__(self,*args,**kwargs):
        self._estimator_type='regressor'
        super().__init__(*args,**kwargs)

def modelfunc(**kwargs):
    for k in DEFAULT_PARAMS:
        if k not in kwargs:
            kwargs[k]=DEFAULT_PARAMS[k]
    lastlayer_activation = 'sigmoid' if kwargs['estimator_type']=='classification' else 'linear'
    mlayers = []
    for lsize in kwargs['layers']:
        if kwargs['pdrop']>0: mlayers.append(tf.keras.layers.Dropout(rate=kwargs['pdrop']))
        mlayers.append(tf.keras.layers.Dense(lsize,activation='relu'))
    mlayers.append(tf.keras.layers.Dense(1,activation=lastlayer_activation))

    model = tf.keras.Sequential(mlayers)
    
    model.compile(optimizer=kwargs['optimizer'],
              loss=kwargs['loss'],
              metrics=['accuracy'])
    
    return model

def get_tf_model(**kwargs):
    mfunc = lambda: modelfunc(**kwargs)
    skmodel = ProperKerasClassifier(mfunc)
    pipe = pipeline.Pipeline([
        ('impute',Imputer()),
        ('scale',StandardScaler()),
        ('model',skmodel)
    ])
    return pipe

class FastKerasClassifier(ProperKerasClassifier):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    def predict_proba(self,X):
        if 'pandas.' in str(type(X)): X = X.values
        rawpreds = self.model(X).numpy().flatten()
        return np.vstack((1-rawpreds,rawpreds)).T

def get_fast_keras(**kwargs):
    mfunc = lambda: modelfunc(**kwargs)
    return FastKerasClassifier(mfunc)
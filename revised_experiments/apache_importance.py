import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from coai import knapsack, base, plots, cegb
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.experimental import enable_iterative_imputer
from sklearn import impute, linear_model
from collections import defaultdict
import itertools, sys
from shap import TreeExplainer, LinearExplainer
from explainers import OneDimExplainer, Loss1DExplainer
from tuning import lintune, tune, bootstrap_set
import dill
from scipy import stats

import config as config

from load_ed import load_ed
from load_eicu import load_eicu, aps_baselines
from load_outpatient import load_outpatient

def prob_regressor(**kwargs):
    return lgb.LGBMRegressor(objective='cross_entropy',**kwargs)

def soft_xent(y,p):
    return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))

def train(rseed):
    (Xtrain,ytrain), (Xvalid,yvalid), (Xtest,ytest), costs, groups, extras = load_eicu(split_seed=rseed)
    
    strain, svalid, stest = [s['apacheiva'].values for s in aps_baselines(split_seed=rseed)]#[extras['apacheiva'][k] for k in ['train','valid','test']]
    
    Xtv = pd.concat([Xtrain,Xvalid])
    ytv = np.hstack((ytrain,yvalid))
    stv = np.hstack((strain,svalid))
    
    model = tune(Xtrain,Xvalid,strain,svalid,mtype=prob_regressor,predfunc=lambda m,X: m.predict(X),scorefunc=lambda y,p: -soft_xent(y,p))
    
    # Print performance
    vbst = prob_regressor(**model)
    vbst.fit(Xtrain,strain)
    print(f'Rank correlation: {stats.spearmanr(stest,vbst.predict(Xtest)).correlation:.3f}')
    
    bst = prob_regressor(**model)
    bst.fit(Xtv,stv)
    exp = OneDimExplainer(bst)
    shaps = exp.shap_values(Xtv)
    global_importance = np.abs(shaps).mean(0)
    np.save(f'eicu_importance_{rseed}.npy',
            np.vstack((np.array(Xtrain.columns).astype('object'),global_importance)))
    
    
    

def main():
    assert len(sys.argv)==2, "USAGE: python apache_importance.py SEED"
    rseed = int(sys.argv[1])
    if rseed<0: rseed = None
    print(f'Running with seed {rseed}...')
    
    train(rseed)
    
    
if __name__ == '__main__': main()
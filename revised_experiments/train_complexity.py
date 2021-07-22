import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import pandas as pd
import lightgbm as lgb
from coai import knapsack, base, plots, cegb, cwcf
from tqdm import tqdm
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.experimental import enable_iterative_imputer
from sklearn import impute, linear_model, preprocessing, compose, pipeline
from collections import defaultdict
import itertools, sys
from shap import TreeExplainer, LinearExplainer
from explainers import OneDimExplainer, Loss1DExplainer, get_sage_wrapper, icu_sage_wrapper, get_pipeline_explainer, labelless_sage_wrapper
import tuning
from tuning import lintune, tune, tftune, bootstrap_set, get_linear_model
from linmodels import FastLinearClassifier
from tfmodels import get_tf_model, ProperKerasClassifier, FastKerasClassifier, get_fast_keras
import dill, pickle

import config as config

from load_ed import load_ed, cost_pair

DEFAULT_GPU=0
BUDGET = config.ED_SURVEY_BUDGET
MAX_ITER = config.COMPLEXITY_MAX_ITER
MTYPES = defaultdict(lambda: knapsack.GroupOptimizer, {
    'fixedmodel': knapsack.FixedModelExactRetainer,  # Always a GBM
    'imputemodel': knapsack.FixedModelImputer,  # Always a GBM
    'cegb': cegb.CEGBOptimizer,  # Always a GBM
    'cwcf': cwcf.CWCFClassifier,  # Always RL
    'cwcflagrange': cwcf.CWCFClassifier,  # Always RL
    'apacheiva': None,  # Clinical score
    'apacheiii': None,  # Clinical score
    'aps': None,  # Clinical score
    'pact': None  # Clinical score
})
VALID_DSETS = set(['trauma'])
LOADERS = {'trauma': lambda **kwargs: load_ed(name=config.ED_NAME,costtype=config.ED_COSTTYPE,drop_redundant=True,**kwargs)}

OUTPATH = config.RUN_PATH
    
# Main training function
def train(mname,rseed,lmbd):
    
    ##################
    # DATA
    ##################
    # Load data we're using
    dname='trauma'
    (Xtrain,ytrain), (Xvalid,yvalid), (Xtest,ytest), costs, groups, extras = LOADERS[dname](split_seed=rseed)#,**kwargs)
            
    # Concatenated data for post-tuning
    Xtv = pd.concat([Xtrain,Xvalid])
    ytv = np.hstack((ytrain,yvalid))

    ##################
    # PARAMETER TUNING
    ##################
    # If we've precomputed best parameters, just load those
    if 'cwcf' not in mname:
        with open(f'{OUTPATH}/gbmsage-{dname}-{rseed}.pkl','rb') as w:
            model = pickle.load(w)
        model['n_jobs']=4
    
    
    ##################
    # Setup for CoAI
    ##################    
    if 'cwcf' not in mname:
        # Instantiate predictive models
        bst = lgb.LGBMClassifier(**model)

        # Get our explainer (using SAGE entirely now, shap is old & may not work perfectly)
        exp = labelless_sage_wrapper(imputetype='marginal',refsize=64,batch_size=32,wrap_categorical=(dname=='icu'))    
    
    lmbds = np.array([lmbd])
    #####################
    # Actually train/test
    #####################
    if 'sage' in mname or 'shap' in mname:
        GRP = knapsack.GroupOptimizer(bst,exp,scale_ints=1000*100 if ('sage' in mname) else 1000)
        GRP.fit(Xtv,ytv,costs,groups,lmbds)
        GRP.score_models_proba(Xtest,ytest,roc_auc_score)
        cost, score = GRP.model_costs[1], GRP.model_scores[1]
    elif mname=='cegb':
        GRP = cegb.CEGBOptimizer(model=bst,lambdas=lmbds)
        GRP.fit(Xtv,ytv,costs)
        GRP.score_models_proba(Xtest,ytest,roc_auc_score)
        cost, score = GRP.model_costs[1], GRP.model_scores[1]
    elif mname=='cwcf':
        ytrain, yvalid, ytest = [np.array(x) for x in (ytrain,yvalid,ytest)]
        print('Training CWCF...')
        GRP = cwcf.get_cwcf(Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,costs,lmbds,
                            gpus=[int(os.environ['CUDA_VISIBLE_DEVICES'])],njobs=1,dirname='/sdata/coai/',
                           metric=roc_auc_score,difficulty=1000)
        print('Done')
    elif mname=='cwcflagrange':
        ytrain, yvalid, ytest = [np.array(x) for x in (ytrain,yvalid,ytest)]
        print('Training CWCF...')
        GRP = cwcf.get_cwcf(Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,costs,lmbds,
                            gpus=[int(os.environ['CUDA_VISIBLE_DEVICES'])],njobs=1,dirname='/sdata/coai/',lagrange=True,
                           metric=roc_auc_score,difficulty=1000)
        print('Done')
        cost, score = GRP.model_costs[0], GRP.model_scores[0]
    else: raise ValueError("Model name not found!")
        
    # Done    
    return cost, score#(GRP.model_costs, GRP.model_scores)
    
# Score models
def train_complexity(mname,rseed):
    target = BUDGET
    best_score = 0.5
    if 'sage' in mname:
        cost, score = train(mname,rseed, target)
        return np.array([target,target]),np.array([0.0,cost]),np.array([0.5,score])
    if 'cwcflagrange' in mname:
        cost, score = train(mname,rseed, target)
        return np.array([target,target]),np.array([0.0,cost]),np.array([0.5,score])
    lmbd, start, end = 1.0, 0, 1e6
    results = []
    
    iterator = tqdm(range(MAX_ITER),desc='Best score:')
    for i in iterator:
        cost, score = train(mname, rseed, lmbd)
        results.append((lmbd,cost,score))
        if cost>target:
            start = lmbd
        else:
            end = lmbd
        lmbd = (start+end)/2
        if (score>best_score) and (cost<=target):
            iterator.set_description(f'Best score: {score:.2f}')
            best_score = score
    return np.array(results).T

def main():
    print(sys.argv)
    assert len(sys.argv)==3, "USAGE: python train_complexity.py MODEL SEED"
    mname = sys.argv[1]
    rseed = int(sys.argv[2])
    
#     assert mname in MTYPES.keys(), f"Valid model names are {list(MTYPES.keys())}"
    
    print(f'Running with dset trauma mtype {mname} seed {rseed}...')
    
    lmbds, costs, scores = train_complexity(mname,rseed)
    stacked = np.vstack([lmbds, costs,scores]).T
    np.save(f'{OUTPATH}/trauma-{mname}-{rseed}-complexity.npy',stacked)
    
    
if __name__ == '__main__': main()
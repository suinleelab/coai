import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn import pipeline
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from tfmodels import get_tf_model, get_fast_keras

import torch
from node.coai_wrapper import NodeClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer

def sub_inds(arr,inds):
    return arr.iloc[inds] if type(arr) in (pd.core.frame.DataFrame,pd.core.series.Series) else arr[inds]
    
def bootstrap_set(*arrays,rseed=None):
    state = np.random.RandomState(seed=rseed)
    a = arrays[0]
    inds = np.arange(a.shape[0]) if rseed is None else state.choice(a.shape[0],size=a.shape[0],replace=True) 
    results = [sub_inds(arr,inds) for arr in arrays]
    return results[0] if len(results)==1 else results

def get_linear_model(**kwargs):
    return pipeline.Pipeline([
        ('impute',Imputer()),
        ('scale',StandardScaler()),
        ('model',LogisticRegression(**kwargs))
    ])
LIN_GRID = {'penalty':['l1','l2'], 'C': np.logspace(-5,5,11),'solver':['saga'],'max_iter':[1000]}
LINTYPE = get_linear_model#LogisticRegression
def lintune(Xtrain,Xvalid,ytrain,yvalid, mfunc=LINTYPE, return_extras=False):
    bestimator = None
    best_score = -float('inf')
    best_params = None
    params_scores = {}
    results = {}
    
    params = LIN_GRID.copy()
    
    lin_keys = params.keys()
    for vals in tqdm(itertools.product(*[params[k] for k in lin_keys])):
        paramdict = {k:v for k,v in zip(lin_keys,vals)}
        results[vals]={}
        model = mfunc(**paramdict)
        model.fit(Xtrain,ytrain)
        cur_score = roc_auc_score(yvalid,model.predict_proba(Xvalid)[:,1])
        params_scores[vals]=cur_score
        if cur_score>best_score:
            bestimator = model
            best_score = cur_score
            best_params = paramdict
    return (best_params, params_scores) if return_extras else best_params

XG_GRID = {
        "learning_rate": [0.01], #[0.001,0.01,0.1],
        "n_estimators":[10,100,1000],
        "num_leaves": [2**x for x in [3,6,9]],#,12]],
        "max_depth":[1,2,4,8],#,16],
        #"gamma": [1.0], #[0.1,1.0,10.0],
        #"min_child_weight": [10], #[1,10,100],
        "subsample":[0.2,0.5,0.8,1.0],
        "n_jobs":[-1],
#         "tree_method":["approx"],
        "verbosity":[-1],
        "silent":[True]
    }
MTYPE = lgb.LGBMClassifier
# Fit params
DEFAULT_PARAMS = dict(verbose=False)#eval_metric='auc',early_stopping_rounds=100
def pred_1dprobs(model,X):
    return model.predict_proba(X)[:,1]
def tune(Xtrain,Xvalid,ytrain,yvalid,mtype=MTYPE,predfunc=pred_1dprobs,scorefunc=roc_auc_score,return_score=False,return_extras=False,**kwargs):
    for k in DEFAULT_PARAMS: 
        if k not in kwargs: 
            kwargs[k] = DEFAULT_PARAMS[k]
    xg_bestimator = None
    xg_best_score = -float('inf')
    xg_best_params = None
    xg_params_scores = {}
    xg_results = {}
    
    params = XG_GRID.copy()
#     params['base_score'] = [np.mean(ytrain)]
    
    xg_keys = params.keys()
    for vals in tqdm(list(itertools.product(*[params[k] for k in xg_keys]))):
        paramdict = {k:v for k,v in zip(xg_keys,vals)}
        xg_results[vals]={}
        bst = mtype(**paramdict)
        bst.fit(Xtrain,ytrain,**kwargs)#,eval_set=[(Xvalid,yvalid)]
        cur_score = scorefunc(yvalid,predfunc(bst,Xvalid))
        xg_params_scores[vals]=cur_score
        if cur_score>xg_best_score:
            xg_bestimator = bst
            xg_best_score = cur_score
            xg_best_params = paramdict
    if not return_extras:
        return (xg_best_params,xg_best_score) if return_score else xg_best_params
    else:
        return ((xg_best_params,xg_best_score),xg_params_scores)

TF_GRID = {
    'layers':[tuple(x) for x in [[16],[32],[64],[128],[256],[512],[64,16],[128,32],[256,64],[256,64,16],[512,128,32]]],
    'pdrop': [0.,0.25,.5,.75],
    'epochs': [10]
    }
def tftune(Xtrain,Xvalid,ytrain,yvalid,mfunc=get_tf_model,scorefunc=roc_auc_score,return_extras=False,iterator=tqdm,**kwargs):
    kwargs = kwargs.copy()
    tf_bestimator = None
    tf_best_score = -float('inf')
    tf_best_params = None
    tf_params_scores = {}
    tf_results = {}
    
    params = TF_GRID.copy()
    
    tf_keys = params.keys()
    for vals in tqdm(itertools.product(*[params[k] for k in tf_keys])):
        paramdict = {k:v for k,v in zip(tf_keys,vals)}
        epochs = paramdict.pop('epochs')
        tf_results[vals]={}
        model = mfunc(**paramdict,**kwargs)
        if 'pipeline' in str(type(model['model'])): fit_params = {'model__model__epochs':epochs, 'model__model__verbose':0}
        else: fit_params = {'model__epochs':epochs, 'model__verbose':0}
        model.fit(Xtrain,ytrain,**fit_params)
        cur_score = scorefunc(yvalid,model.predict_proba(Xvalid)[:,1])
        tf_params_scores[vals]=cur_score
        if cur_score>tf_best_score:
            tf_bestimator = model
            tf_best_score = cur_score
            tf_best_params = paramdict
    return ((tf_best_params,tf_best_score),tf_params_scores) if return_extras else tf_best_params


DEFAULT_TAB = dict( n_independent=2, n_shared=2,
    cat_idxs=[],
    cat_dims=[],
    cat_emb_dim=1,momentum=0.3, clip_value=2.,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params = {"gamma": 0.95,
                     "step_size": 20},
    scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15, verbose=0)
TUNE_TAB = dict(n_d_a=[8,16,32,64],
                n_steps=[2,4,6,8],
                gamma=[1.1,1.3,1.5,1.7,1.9],
                lambda_sparse=[1e-6,1e-4,1e-2,1e0,1e1])
MAX_EPOCHS=1000

def tabtune(Xtrain,Xvalid,ytrain,yvalid,
            verbose=True,
            scorefunc=roc_auc_score,
            predfunc=pred_1dprobs,return_extras=False,
            return_score=False,
            **kwargs):
    for k in DEFAULT_TAB: 
        if k not in kwargs: 
            kwargs[k] = DEFAULT_TAB[k]
    xg_bestimator = None
    xg_best_score = -float('inf')
    xg_best_params = None
    xg_params_scores = {}
    xg_results = {}
    
    params = TUNE_TAB.copy()
    
    xg_keys = params.keys()
    for vals in tqdm(list(itertools.product(*[params[k] for k in xg_keys]))):
        paramdict = {k:v for k,v in zip(xg_keys,vals)}
        n_d_a = paramdict.pop('n_d_a')
        paramdict['n_d'] = paramdict['n_a'] = n_d_a
        xg_results[vals]={}
        bst = TabNetClassifier(**kwargs,**paramdict)
        bst.fit(Xtrain,ytrain,
                eval_set=[(Xtrain,ytrain),(Xvalid,yvalid)],
               eval_metric=['auc'],
               max_epochs=MAX_EPOCHS,
               patience=20, batch_size=1024,virtual_batch_size=128,num_workers=0,weights=1,drop_last=False)#,eval_set=[(Xvalid,yvalid)]
        cur_score = scorefunc(yvalid,predfunc(bst,Xvalid))
        xg_params_scores[vals]=cur_score
        if cur_score>xg_best_score:
            xg_bestimator = bst
            xg_best_score = cur_score
            xg_best_params = paramdict
#     xg_best_params['n_estimators']=bst.booster_.best_iteration
    if not return_extras:
        return (xg_best_params,xg_best_score) if return_score else xg_best_params
    else:
        return ((xg_best_params,xg_best_score),xg_params_scores)

    
TUNE_NODE = dict(num_trees=[512],
                num_layers=[2,4,8],
                depth=[6],
                tree_dim=[2,3])
MAX_EPOCHS=1000

def nodetune(Xtrain,Xvalid,ytrain,yvalid,
            verbose=True,
            scorefunc=roc_auc_score,
             mfunc=NodeClassifier,
            predfunc=pred_1dprobs,return_extras=False,
            return_score=False,
            **kwargs):
    xg_bestimator = None
    xg_best_score = -float('inf')
    xg_best_params = None
    xg_params_scores = {}
    xg_results = {}
    
    params = TUNE_NODE.copy()
    
    xg_keys = params.keys()
    for i,vals in tqdm(list(enumerate(itertools.product(*[params[k] for k in xg_keys])))):
        paramdict = {k:v for k,v in zip(xg_keys,vals)}
        paramdict['num_trees'] //= (paramdict['num_layers']//2)
        xg_results[vals]={}
        bst = mfunc(experiment_name=f'nodetune_{id(paramdict)}_{i}',**kwargs,**paramdict)
        bst.fit(Xtrain,ytrain,
                eval_set=(Xvalid,yvalid))
        cur_score = scorefunc(yvalid,predfunc(bst,bst.dataset.X_valid))
        xg_params_scores[vals]=cur_score
        if cur_score>xg_best_score:
            xg_bestimator = bst
            xg_best_score = cur_score
            xg_best_params = paramdict
    if not return_extras:
        return (xg_best_params,xg_best_score) if return_score else xg_best_params
    else:
        return ((xg_best_params,xg_best_score),xg_params_scores)
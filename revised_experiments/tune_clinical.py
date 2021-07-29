import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import pandas as pd
import lightgbm as lgb
from coai import knapsack, base, plots, cegb, cwcf
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.experimental import enable_iterative_imputer
from sklearn import impute, linear_model, preprocessing, compose, pipeline
from collections import defaultdict
import itertools, sys
from shap import TreeExplainer, LinearExplainer
from explainers import OneDimExplainer, Loss1DExplainer, get_sage_wrapper, icu_sage_wrapper, get_pipeline_explainer, labelless_sage_wrapper
import tuning
from tuning import lintune, tune, tftune, bootstrap_set, get_linear_model, tabtune, nodetune
from node.coai_wrapper import NodeClassifier
from linmodels import FastLinearClassifier
from tfmodels import get_tf_model, ProperKerasClassifier, FastKerasClassifier, get_fast_keras
import dill, pickle
import tensorflow as tf

import config as config

from load_ed import load_ed, cost_pair
from load_eicu import load_eicu, aps_baselines
from load_outpatient import load_outpatient
from baselines import pact_score
from qsofa import qsofa_score

SAVE_MODELS = False
TUNING = 'SAVE' #Options ['SAVE' (Hyperparam tune then save and exit),'LOAD' (Load cached hyperparams), 'ANYTHING ELSE' (Tunes, saves, and continues)]

MTYPES = defaultdict(lambda: knapsack.GroupOptimizer, {
    'fixedmodel': knapsack.FixedModelExactRetainer,  # Always a GBM
    'imputemodel': knapsack.FixedModelImputer,  # Always a GBM
    'cegb': cegb.CEGBOptimizer,  # Always a GBM
    'cwcf': cwcf.CWCFClassifier,  # Always RL
    'apacheiva': None,  # Clinical score
    'apacheiii': None,  # Clinical score
    'aps': None,  # Clinical score
    'pact': None  # Clinical score
})
VALID_DSETS = set(['trauma','icu','outpatient'])
LOADERS = {'trauma': lambda **kwargs: load_ed(name=config.ED_NAME,costtype=config.ED_COSTTYPE,drop_redundant=True,**kwargs), 
           'icu': load_eicu, 'outpatient': load_outpatient}

OUTPATH = config.RUN_PATH

# Introduces nswap swaps into cost vector for "swap robustness"
def cost_swaps(costs,nswap,seed=None):
    rng= np.random.RandomState(seed)
    newcosts = costs.copy()
    for i in range(nswap):
        inds = rng.choice(newcosts.shape[0],2)
        newcosts[inds] = newcosts[inds[::-1]]
    return newcosts

# Preprocess ICU data by string-encoding NaNs and onehot encoding admission diagnosis
def icu_preprocessing(mfunc):
    return lambda **kwargs: pipeline.Pipeline([
    ('fillna', compose.ColumnTransformer([
        ('nanstring',impute.SimpleImputer(strategy='constant',fill_value='NaN'),['admitdiagnosis'])
    ],remainder='passthrough')),
    # Have to hackily encode column as 0 on second transformer bc columntransformer throws out Pandas info
    ('ohe', compose.ColumnTransformer([
            ('onehot',preprocessing.OneHotEncoder(sparse=False,handle_unknown='ignore'),[0])
    ],remainder='passthrough')),
    ('impute',impute.SimpleImputer()),
    ('scale',preprocessing.StandardScaler()),
    ('model',mfunc(**kwargs))
])
    
# Main training function
def train(dname,mname,rseed, shuffle_params=None):
    assert (('gbm' in mname) or (mname in ('cegb','fixedmodel','imputemodel')) or ('linear' in mname) or ('nn' in mname) or ('tab' in mname) or ('node' in mname))
    
    ##################
    # DATA
    ##################
    # Load data we're using
    (Xtrain,ytrain), (Xvalid,yvalid), (Xtest,ytest), costs, groups, extras = LOADERS[dname](split_seed=rseed)#,**kwargs)
    
    # If we're using PACT we need some of the extra (redundant) features that were unused in our study
    if mname=='pact': (Xtrain,ytrain), (Xvalid,yvalid), (Xtest,ytest), costs, groups, extras = load_ed(name=config.ED_NAME,costtype=config.ED_COSTTYPE,drop_redundant=False,split_seed=rseed)
    
    
    # If we're using a non-GBM AI method, we need to impute NaNs and scale
    # Don't do this if using ICU data because we're using a Pipeline in that case
    # that handle sthis stuff
    if ('linear' in mname or 'nn' in mname or 'cwcf' in mname or 'tab' in mname or 'node' in mname) and (dname!='icu'):
        imputer = impute.SimpleImputer()
        scaler = preprocessing.StandardScaler()
        Xtrain_np = scaler.fit_transform(imputer.fit_transform(Xtrain))
        Xvalid_np = scaler.transform(imputer.transform(Xvalid))
        Xtest_np = scaler.transform(imputer.transform(Xtest))

        for df, npy in zip([Xtrain,Xvalid,Xtest],[Xtrain_np,Xvalid_np,Xtest_np]):
            df.iloc[:] = npy
#         Xtrain,Xvalid,Xtest = [pd.DataFrame(data=npy,columns=df.columns,index=df.index) for df,npy in zip(
#             [Xtrain_raw,Xvalid_raw,Xtest_raw],[Xtrain_np,Xvalid_np,Xtest_np])]
#     else:
#         (Xtrain,Xvalid,Xtest) = Xtrain_raw,Xvalid_raw,Xtest_raw
    
    # Concatenated data for post-tuning
    Xtv = pd.concat([Xtrain,Xvalid])
    ytv = np.hstack((ytrain,yvalid))
    
    # Grouped costs for datasets tht feature it
    unique_costs = np.array([costs[groups==g].mean() for g in np.unique(groups)]) if (dname=='outpatient') or (dname=='icu' and mname in ('linear','linearh','nn')) else costs

    ##################
    # PARAMETER TUNING
    ##################
    # If we've precomputed best parameters, just load those
    if TUNING=='LOAD' and (('gbm' in mname) or (mname in ('cegb','fixedmodel','imputemodel')) or ('linear' in mname) or ('nn' in mname) or ('tab' in mname)):
        loadname = 'gbmsage' if mname=='cegb' else mname
        with open(f'{OUTPATH}/{loadname}-{dname}-{rseed}.pkl','rb') as w:
            model = pickle.load(w)
    # Otherwise do some parameter tuning
    else:
        # Tune GBM
        if ('gbm' in mname) or (mname in ('cegb','fixedmodel','imputemodel')):
            model = tune(Xtrain,Xvalid,ytrain,yvalid)
        # Linear model needs onehotencoding pipeline if we're doing ICU
        elif ('linear' in mname):
            if (dname=='icu'):
                model = lintune(Xtrain,Xvalid,ytrain,yvalid,mfunc=icu_preprocessing(get_linear_model))
            else:
                model = lintune(Xtrain,Xvalid,ytrain,yvalid)
        # NN model needs onehotencoding pipeline if we're doing ICU
        elif 'nn' in mname:
            if (dname=='icu'):
                model = tftune(Xtrain,Xvalid,ytrain,yvalid,mfunc=icu_preprocessing(get_tf_model),return_extras=False)
            else:
                model = tftune(Xtrain,Xvalid,ytrain,yvalid,return_extras=False)
        elif 'node' in mname: 
            model = nodetune(Xtrain,Xvalid,ytrain,yvalid,
                            mfunc=(icu_preprocessing(NodeClassifier) if dname=='icu' else NodeClassifier))
            if dname!='icu':
                bst = NodeClassifier(**model)
                bst.fit(Xtrain,ytrain,eval_set=(Xvalid,yvalid))
                iXtest = bst.dataset.transform(Xtest)
                preds = bst.predict_proba(iXtest)[:,1]
                score = roc_auc_score(ytest,preds)
                model['test_score']=score
        elif ('tab' in mname):
            cat_name_map = {'trauma':['agencylevelfromscene','agencymodefromscene','ageunits','causecode','ethnicity','formfromscene','race','residencestate','scenedestinationreason','scenerespassisted','sex']
}
            cat_idx_map = {'trauma':[i for i,c in enumerate(Xtrain.columns) if c in 
                                    cat_name_map['trauma']]}
            cat_dim_map = {'trauma':[Xtrain[c].unique().shape[0] for i,c in enumerate(Xtrain.columns) if c in 
                                    cat_name_map['trauma']]}
            if (dname=='icu'):
                model = tabtune(Xtrain.values,Xvalid.values,ytrain,yvalid,mfunc=icu_preprocessing(get_linear_model))
            else:
                model = tabtune(Xtrain.values,Xvalid.values,ytrain,yvalid,
                               cat_idxs=cat_name_map.get(dname,[]),
                               cat_dims=cat_dim_map.get(dname,[]),
                               cat_emb_dim=2,return_score=True)
    # If we indicated we want to save the model, do so
    if TUNING=='SAVE' and (('gbm' in mname) or (mname in ('cegb','fixedmodel','imputemodel')) or ('linear' in mname) or ('nn' in mname) or ('tab' in mname) or ('node' in mname)):
        with open(f'{OUTPATH}/{mname}-{dname}-{rseed}.pkl','wb') as w:
            pickle.dump(model,w)
            exit()
    
    
    ##################
    # Setup for CoAI
    ##################    
    # Instantiate predictive models
    if ('gbm' in mname) or (mname in ('cegb','fixedmodel','imputemodel')):
        bst = lgb.LGBMClassifier(**model)
    elif 'linear' in mname:
        bst = icu_preprocessing(FastLinearClassifier)(**model) if dname=='icu' else FastLinearClassifier(**model)
    elif 'nn' in mname:
        bst = icu_preprocessing(get_fast_keras)(**model) if dname=='icu' else get_fast_keras(**model)
    
    # Get our explainer (using SAGE entirely now, shap is old & may not work perfectly)
    if ('sage' in mname) or (mname in ('cegb','fixedmodel','imputemodel')):
        exp = labelless_sage_wrapper(imputetype='marginal',refsize=64,batch_size=32,wrap_categorical=(dname=='icu'))
    elif mname=='gbmshap':
        exp = OneDimExplainer
    elif mname=='linearshap':
        exp = get_pipeline_explainer(LinearExplainer)
        
    # Prepare to shuffle costs if required
    if shuffle_params is not None: 
        if ((shuffle_params[0]<0) and (shuffle_params[1]<0)):
            costs, shuffle_costs = cost_pair(-shuffle_params[0],-shuffle_params[1],Xtrain)
        else:
            shuffle_costs = cost_swaps(costs,shuffle_params[0],shuffle_params[1])
    # Pick thresholds for CoAI    
    dthresh = np.linspace(0,np.sum(unique_costs)+1,100)
    
    
    
    #####################
    # Actually train/test
    #####################
    if 'sage' in mname or 'shap' in mname:
        GRP = knapsack.GroupOptimizer(bst,exp,scale_ints=1000*100 if ('sage' in mname) else 1000)
        if 'nn' in mname: 
            if dname=='icu': GRP.fit(Xtv,ytv,costs,groups,dthresh,model__epochs=10,model__verbose=False)
            else: GRP.fit(Xtv,ytv,costs,groups,dthresh,epochs=10,verbose=False)
        else: GRP.fit(Xtv,ytv,costs,groups,dthresh)
        GRP.score_models_proba(Xtest,ytest,roc_auc_score)
        if shuffle_params: GRP.recalculate_costs(shuffle_costs)
    elif 'fixed' in mname:
        bst = bst.fit(Xtv,ytv)
        GRP = knapsack.FixedModelExactRetainer(bst,exp)
        GRP.fit(Xtv,ytv,costs,dthresh)
        if shuffle_params: GRP.refit(Xtv,ytv,shuffle_costs)
        GRP.score_models_proba(Xtest,ytest,roc_auc_score)
    elif 'impute' in mname:
        imputer = impute.IterativeImputer(random_state=0,estimator=linear_model.RidgeCV())
        bst = bst.fit(Xtv,ytv)
        imputer.fit(Xtv)
        GRP = knapsack.FixedModelImputer(bst,exp,imputer)
        GRP.fit(Xtv,ytv,costs,dthresh)
        if shuffle_params: GRP.refit(Xtv,ytv,shuffle_costs)
        GRP.score_models_proba(Xtest,ytest,roc_auc_score)
    elif mname=='cegb':
        GRP = cegb.CEGBOptimizer(model=bst,lambdas=np.logspace(-5,5,101))
        GRP.fit(Xtv,ytv,costs)
        GRP.score_models_proba(Xtest,ytest,roc_auc_score)
        if dname=='outpatient': GRP.recalculate_costs(costs,groups)
        if shuffle_params: GRP.recalculate_costs(shuffle_costs)
    elif mname=='cwcf':
        ytrain, yvalid, ytest = [np.array(x) for x in (ytrain,yvalid,ytest)]
        print('Training CWCF...')
        lmbds = np.hstack([np.logspace(-14,1,16) for _ in range(2)])
        GRP = cwcf.get_cwcf(Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,costs,lmbds,
                            gpus=list(range(8)),njobs=32,dirname=config.CWCF_TMPDIR,
                           metric=roc_auc_score,difficulty=1000)
        print('Done')
    elif mname in ('aps','apacheiii','apacheiva'):
        strain,svalid,stest = aps_baselines()
        mpreds = stest
        mpreds = bootstrap_set(mpreds,rseed=rseed)
        preds = mpreds[mname]
        score = roc_auc_score(ytest,preds)
        cost = config.EICU_SCORE_COSTS[mname]
        GRP = lambda x: x
        GRP.model_costs, GRP.model_scores = np.array([cost]), np.array([score])
        GRP.test_preds = np.array(preds)
    elif mname in ('qsofa'):
        qtest = qsofa_score()
        qpreds = bootstrap_set(qtest,rseed=rseed)
        score = roc_auc_score(ytest,qpreds)
        cost = config.EICU_SCORE_COSTS[mname]
        GRP = lambda x: x
        GRP.model_costs, GRP.model_scores = np.array([cost]), np.array([score])
        GRP.test_preds = np.array(qpreds)
    elif mname in ('pact'):
        cost, score, _, _, _, _, preds = pact_score(Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,costs)
        GRP = lambda x: x
        GRP.model_costs, GRP.model_scores = np.array(cost), np.array(score)
        GRP.test_preds = np.array(preds)
    else: raise ValueError("Model name not found!")
        
    # Done    
    return GRP
    
# Score models
def train_costperf(dname,mname,rseed,shuffle_params):
    model = train(dname,mname,rseed,shuffle_params)
    if SAVE_MODELS:
        if hasattr(model,'base_explainer'): del model.base_explainer
        if 'nn' in mname: del model.models
        with open(f'{OUTPATH}/{dname}-{mname}-{rseed}-{shuffle_params}.coai','wb') as w:
            dill.dump(model,w)
    return model.model_costs, model.model_scores

def main():
    print(sys.argv)
    assert len(sys.argv)>=4, "USAGE: python train_clinical.py DATA MODEL SEED NSWAPS SWAPSEED"
    dname = sys.argv[1]
    mname = sys.argv[2]
    rseed = int(sys.argv[3])
    if len(sys.argv)>4:
        nswap = int(sys.argv[4])
        swapseed = int(sys.argv[5])
#         if swapseed<0: swapseed=None
        shuffle_params = (nswap,swapseed)
    else: shuffle_params = None
    if rseed<0: rseed = None
    assert dname in VALID_DSETS, "Valid dataset names are (trauma, icu, outpatient)!"
    
    print(f'Running with dset {dname} mtype {mname} seed {rseed} shuffle params {shuffle_params}...')
    
    costs, scores = train_costperf(dname,mname,rseed,shuffle_params)
    stacked = np.vstack([costs,scores]).T
    np.save(f'{OUTPATH}/{dname}-{mname}-{rseed}-{shuffle_params}.npy',stacked)
    

def test_cegb_error():
    costs, scores = train_costperf('trauma','cegb',92,None)
    
if __name__ == '__main__': main()
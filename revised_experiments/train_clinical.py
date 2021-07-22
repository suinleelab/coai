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
from explainers import OneDimExplainer, OneDimDependent, Loss1DExplainer, get_sage_wrapper, icu_sage_wrapper, get_pipeline_explainer, labelless_sage_wrapper
import tuning
from tuning import lintune, tune, tftune, bootstrap_set, get_linear_model
from linmodels import FastLinearClassifier
from tfmodels import get_tf_model, ProperKerasClassifier, FastKerasClassifier, get_fast_keras
import dill, pickle
import tensorflow as tf
from node.coai_wrapper import NodeClassifier

import config as config

from load_ed import load_ed, cost_pair
from load_eicu import load_eicu, aps_baselines
from load_outpatient import load_outpatient
from baselines import pact_score
from qsofa import qsofa_score

SAVE_MODELS = True
TUNING = 'LOAD' #Options ['SAVE' (Hyperparam tune then save and exit),'LOAD' (Load cached hyperparams), 'ANYTHING ELSE' (Tunes, saves, and continues)]

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
#     ICU preprocessigng is now in its own function
#     mtype = MTYPES[mname]
#     kwargs = {}
#     if dname=='icu' and ('linear' in mname or 'nn' in mname or 'cwcf' in mname): kwargs['onehot']=True
    
#     CWCF now runs in parallel across several GPUs
#     if mname=='cwcf' and 'CUDA_VISIBLE_DEVICES' not in os.environ:
#         ngpu = len(tf.config.list_physical_devices('GPU'))
#         cur_gpu = rseed%ngpu if rseed is not None else 0
#         os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#         os.environ["CUDA_VISIBLE_DEVICES"]=str(cur_gpu)
    
    ##################
    # DATA
    ##################
    # Load data we're using
    (Xtrain,ytrain), (Xvalid,yvalid), (Xtest,ytest), costs, groups, extras = LOADERS[dname](split_seed=rseed)#,**kwargs)
    
    # If we're using PACT we need some of the extra (redundant) features that were unused in our study
    if mname=='pact': (Xtrain,ytrain), (Xvalid,yvalid), (Xtest,ytest), costs, groups, extras = load_ed(name=config.ED_NAME,costtype=config.ED_COSTTYPE,drop_redundant=False,split_seed=rseed)
#     print([(n,c) for n,c in zip(Xtrain.columns,costs) if c>0.01])
    
    
#     Xtrain_raw,Xvalid_raw,Xtest_raw = Xtrain,Xvalid,Xtest
    
#     For bootstrapping, we don't do this anymore and do train/test splits instead
#     Xtrain_raw, ytrain = bootstrap_set(Xtrain,ytrain,rseed=rseed)
#     Xvalid_raw, yvalid = bootstrap_set(Xvalid,yvalid,rseed=rseed)
#     Xtest_raw, ytest = bootstrap_set(Xtest,ytest,rseed=rseed)
    
    # If we're using a non-GBM AI method, we need to impute NaNs and scale
    # Don't do this if using ICU data because we're using a Pipeline in that case
    # that handles this stuff
    if ('linear' in mname or 'nn' in mname or 'cwcf' in mname or 'node' in mname) and (dname!='icu'):
        imputer = impute.SimpleImputer()
        scaler = preprocessing.StandardScaler()
        Xtrain_np = scaler.fit_transform(imputer.fit_transform(Xtrain))
        Xvalid_np = scaler.transform(imputer.transform(Xvalid))
        Xtest_np = scaler.transform(imputer.transform(Xtest))

        for df, npy in zip([Xtrain,Xvalid,Xtest],[Xtrain_np,Xvalid_np,Xtest_np]):
            df.iloc[:] = npy
            
#             Hackier code for preprocessing features, can probably remove
#         Xtrain,Xvalid,Xtest = [pd.DataFrame(data=npy,columns=df.columns,index=df.index) for df,npy in zip(
#             [Xtrain_raw,Xvalid_raw,Xtest_raw],[Xtrain_np,Xvalid_np,Xtest_np])]
#     else:
#         (Xtrain,Xvalid,Xtest) = Xtrain_raw,Xvalid_raw,Xtest_raw
    
    # Concatenated data for training cost-aware models after tuning
    Xtv = pd.concat([Xtrain,Xvalid])
    ytv = np.hstack((ytrain,yvalid))
    
    # Grouped costs for datasets tht feature it
    # Outpatient dataset
    # Or linear/NN on ICU (one-hot encoding of admission dx)
    unique_costs = np.array([costs[groups==g].mean() for g in np.unique(groups)]) if (dname=='outpatient') or (dname=='icu' and mname in ('linear','linearh','nn')) else costs

    ##################
    # PARAMETER TUNING
    ##################
    # If we've precomputed best parameters, just load those
    if TUNING=='LOAD' and (('gbm' in mname) or (mname in ('fixedmodel','imputemodel')) or ('linear' in mname) or ('nn' in mname) or ('cegb' in mname)):
        loadname = 'gbmsage' if ((mname in ('fixedmodel','imputemodel')) or ('cegb' in mname) or ('gbmsage' in mname)) else mname
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
        # NODE model doesn't need tuning
        elif 'node' in mname:
            model = {}
    # If we indicated we want to save the model, do so
#     print(model)
    if TUNING=='SAVE' and (('gbm' in mname) or (mname in ('cegb','fixedmodel','imputemodel')) or ('linear' in mname) or ('nn' in mname)):
        with open(f'{OUTPATH}/{mname}-{dname}-{rseed}.pkl','wb') as w:
            pickle.dump(model,w)
            exit()
            
    # Limit number of jobs for processor-hungry models
    print(mname)
    if mname not in ('qsofa','aps','apacheiii','apacheiva'):
        if (('gbm' in mname) or ('cegb' in mname) or ('linear' in mname) or ('imputemodel' in mname)): 
            model['n_jobs']= 4 if dname=='trauma' else 2
#             else:  model['n_jobs']=10     
    
    ##################
    # Setup for CoAI
    ##################    
    # Instantiate predictive models
    if ('gbm' in mname) or ('cegb' in mname) or (mname in ('fixedmodel','imputemodel')):
        bst = lgb.LGBMClassifier(**model)
    elif 'linear' in mname:
        bst = icu_preprocessing(FastLinearClassifier)(**model) if dname=='icu' else FastLinearClassifier(**model)
    elif 'nn' in mname:
        bst = icu_preprocessing(get_fast_keras)(**model) if dname=='icu' else get_fast_keras(**model)
    elif 'node' in mname:
        bst = icu_preprocessing(NodeClassifier)(experiment_name=f'trauma{rseed}',**model) if dname=='icu' else NodeClassifier(experiment_name=f'trauma{rseed}',**model)
    
    # Get our explainer (using SAGE entirely now, shap code is old & may not work perfectly)
    if ('sage' in mname) or (mname in ('cegb','fixedmodel','imputemodel')):
        #sage_params={'imputetype':'default'}
        #if 'gbm' in mname: sage_params={'imputetype':'marginal'}
        
        # SAGE explainer. N_permutations set super low for NODE bc we're
        # just testing it right now
        exp = labelless_sage_wrapper(imputetype='marginal',refsize=64,batch_size=32,wrap_categorical=(dname=='icu'),n_permutations=(128 if 'node' in mname else None))
        
#         NODE debugging line
#         print(dict(imputetype=('default' if 'node' in mname else 'marginal'),refsize=(1 if 'node' in mname else 64)))

#     Mostly deprecated
    elif mname=='gbmshap':
        exp = OneDimExplainer
    elif mname=='linearshap':
        exp = get_pipeline_explainer(LinearExplainer)
        
    # Prepare to perturb costs if required (robustness experiments)
    if shuffle_params is not None: 
        # Negative numbers indicate individiual robustness
        if ((shuffle_params[0]<0) and (shuffle_params[1]<0)):
            costs, shuffle_costs = cost_pair(-shuffle_params[0],-shuffle_params[1],Xtrain)
        # Positive indicate swap robustness - # swaps and seed
        else:
            shuffle_costs = cost_swaps(costs,shuffle_params[0],shuffle_params[1])
    # Pick thresholds for CoAI    
    dthresh = np.linspace(0,np.sum(unique_costs)+1,100)
    
    
    
    #####################
    # Actually train/test
    #####################
    if 'sage' in mname or 'shap' in mname:
        # Wrap model with CoAI
        if 'greedy' in mname:
            GRP = knapsack.GroupGreedy(bst,exp)
        else:
            GRP = knapsack.GroupOptimizer(bst,exp,scale_ints=1000*100 if ('sage' in mname) else 1000)
        # NN needs preprocessing pipeline if ICU, also pass # epochs, verbosity
        if 'nn' in mname: 
            if dname=='icu': GRP.fit(Xtv,ytv,costs,groups,dthresh,model__epochs=10,model__verbose=False)
            else: GRP.fit(Xtv,ytv,costs,groups,dthresh,epochs=10,verbose=False)
        # NODE needs preprocessing for ICU. 
        # Also requires eval set for stopping time
        # Current max_iter is short for prototyping
        elif 'node' in mname: 
            dthresh = np.linspace(0,np.sum(unique_costs)+1,10)
            if dname=='icu': GRP.fit(Xtrain,ytrain,costs,groups,dthresh,model__eval_set=(Xvalid,yvalid),model__max_iter=15)
            else: GRP.fit(Xtrain,ytrain,costs,groups,dthresh,eval_set=(Xvalid,yvalid),max_iter=15)
        # All other CoAI methods get a standardized fit process
        else: GRP.fit(Xtv,ytv,costs,groups,dthresh)
        # Evaluate CoAI models
        GRP.score_models_proba(Xtest,ytest,roc_auc_score)
        # If costs get shuffled, each model's deployment cost will change
        if shuffle_params: GRP.recalculate_costs(shuffle_costs)
            
    # Impute-CoAI with mean imputation
    elif 'fixed' in mname:
        bst = bst.fit(Xtv,ytv)
        GRP = knapsack.FixedModelExactRetainer(bst,exp)
        GRP.fit(Xtv,ytv,costs,dthresh)
        if shuffle_params: GRP.refit(Xtv,ytv,shuffle_costs)
        GRP.score_models_proba(Xtest,ytest,roc_auc_score)
    # Impute-CoAI with model-based imputation (IterativeImputer)
    elif 'impute' in mname:
        imputer = impute.IterativeImputer(random_state=0,estimator=linear_model.RidgeCV())
        bst = bst.fit(Xtv,ytv)
        imputer.fit(Xtv)
        GRP = knapsack.FixedModelImputer(bst,exp,imputer)
        GRP.fit(Xtv,ytv,costs,dthresh)
        if shuffle_params: GRP.refit(Xtv,ytv,shuffle_costs)
        GRP.score_models_proba(Xtest,ytest,roc_auc_score)
#     GRP.fit(Xtv,ytv,costs,groups,dthresh) if mname=='default' else GRP.fit(Xtv,ytv,costs,dthresh)
    # CEGB doesn't use an explainer
    elif ('cegb' in mname):
        GRP = cegb.CEGBOptimizer(model=bst,lambdas=np.logspace(-5,5,101))
        GRP.fit(Xtv,ytv,costs,groups=(groups if 'group' in mname else None))
        GRP.score_models_proba(Xtest,ytest,roc_auc_score)
        # Account for grouped costs if in outpatient data
        if (dname=='outpatient'): GRP.recalculate_costs(costs,groups)
        # Account for any cost perturbations
        if shuffle_params: GRP.recalculate_costs(shuffle_costs)
    elif ('cwcf' in mname):
        # Lots of preprocessing if using ICU data to encode categoricals
        # as ordinal ints (save memory, handle groups, etc)
        if dname=='icu':
            types = Xtrain.dtypes
            for col in Xtrain.columns:
                if str(types[col]) == 'category':
                    l_enc = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan)
                    for df in [Xtrain,Xvalid,Xtest]:
                        if 'UNK' not in df[col].cat.categories:
                            df[col].cat.add_categories(['UNK'],inplace=True)
                        df[col].fillna('UNK',inplace=True)
                    Xtrain[col] = l_enc.fit_transform(np.array(Xtrain[col]).reshape(-1,1))
                    Xvalid[col] = l_enc.transform(np.array(Xvalid[col]).reshape(-1,1))
                    Xtest[col] = l_enc.transform(np.array(Xtest[col]).reshape(-1,1))

#             Old mode imputation code, better now (broken by dtype)
            #         for df in [Xtrain,Xvalid,Xtest]:
            #             if df[col].isna().any():
            #                 df[col][df[col].isna()] = Xtrain[col].mode().iloc[0]
#                     Xtrain[col] = Xtrain[col].fillna(Xtrain[col].mode().iloc[0])
#                     Xvalid[col] = Xvalid[col].fillna(Xtrain[col].mode().iloc[0])
#                     Xtest[col] = Xtest[col].fillna(Xtrain[col].mode().iloc[0])
                elif str(types[col])=='int64':
                    Xtrain[col].fillna(Xtrain[col].mode(), inplace=True)
                    Xvalid[col].fillna(Xtrain[col].mode(), inplace=True)
                    Xtest[col].fillna(Xtrain[col].mode(), inplace=True)
                else:
                    Xtrain[col].fillna(Xtrain[col].mean(), inplace=True)
                    Xvalid[col].fillna(Xtrain[col].mean(), inplace=True)
                    Xtest[col].fillna(Xtrain[col].mean(), inplace=True)
            
        # CWCF only takes nparrays for labels
        ytrain, yvalid, ytest = [np.array(x) for x in (ytrain,yvalid,ytest)]
        print('Training CWCF...') # So we know when jobs get farmed out to other processes
        # Used to turn "groups" down to 6 for outpatient just to prototype group support
        if 'lagrange' in mname:
            data_lmbds = {'trauma': np.linspace(0,np.sum(unique_costs),17)[1:], 'icu': np.linspace(0,np.sum(unique_costs),17)[1:], 'outpatient': np.linspace(0,np.sum(unique_costs),17)[1:]}
        else:
            data_lmbds = {'trauma': np.logspace(-14,1,16), 'icu': np.logspace(-14,1,16), 'outpatient': np.logspace(-14,1,16)}
        # This is usually range(2) to get some stability over reps -- doesn't matter as much for outpatient
        # Can turn down to 1 when prototyping
        lmbds = np.hstack([data_lmbds[dname] for _ in range(2)])
        # Old single threaded mode
#         GRP = cwcf.CWCFClassifier(costs=costs,dirname=config.CWCF_TMPDIR)
#         GRP.fit(Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest)
#         print([x.shape for x in (Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,costs,lmbds)])

# Run CWCF - groups argument does experimental groups handling (not working yet)
# More jobs (even more than GPUs) can be used - gets you through the lambda list faster
# Set up right now for L3 gpus 1-6.
        GRP = cwcf.get_cwcf(Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,costs,lmbds,
                            gpus=np.random.permutation(8),njobs=16,dirname=config.CWCF_TMPDIR,lagrange=('lagrange' in mname),
                           metric=roc_auc_score,difficulty=1000,groups=(groups if 'group' in mname else None))
        print('Done')  # Done with external process run
    # ICU baselines
    elif mname in ('aps','apacheiii','apacheiva'):
        strain,svalid,stest = aps_baselines(split_seed=rseed)
        mpreds = stest
#         mpreds = bootstrap_set(mpreds,rseed=rseed)
        preds = mpreds[mname]
        score = roc_auc_score(ytest,preds)
        cost = config.EICU_SCORE_COSTS[mname]
        GRP = lambda x: x
        GRP.model_costs, GRP.model_scores = np.array([cost]), np.array([score])
        GRP.test_preds = np.array(preds)
    elif mname in ('qsofa'):
        qtest = qsofa_score(split_seed=rseed)
        qpreds = qtest#bootstrap_set(qtest,rseed=rseed)
        score = roc_auc_score(ytest,qpreds)
        cost = config.EICU_SCORE_COSTS[mname]
        GRP = lambda x: x
        GRP.model_costs, GRP.model_scores = np.array([cost]), np.array([score])
        GRP.test_preds = np.array(qpreds)
    # Trauma baseline (PACT)
    # Should ignore the resulting cost for now and just use 
    # the hand-calculated one
    elif mname in ('pact'):
        cost, score, _, _, _, _, preds = pact_score(Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,costs)
        GRP = lambda x: x
        GRP.model_costs, GRP.model_scores = np.array(cost), np.array(score)
        GRP.test_preds = np.array(preds)
    else: raise ValueError("Model name not found!")
        
    # Done    
    return GRP#(GRP.model_costs, GRP.model_scores)
    
# Score models
def train_costperf(dname,mname,rseed,shuffle_params):
    model = train(dname,mname,rseed,shuffle_params)
    if SAVE_MODELS or (mname=='gbmsage' and shuffle_params is None) or ('pact' in mname):
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
#     assert mname in MTYPES.keys(), f"Valid model names are {list(MTYPES.keys())}"
    
    print(f'Running with dset {dname} mtype {mname} seed {rseed} shuffle params {shuffle_params}...')
    
    costs, scores = train_costperf(dname,mname,rseed,shuffle_params)
    stacked = np.vstack([costs,scores]).T
    np.save(f'{OUTPATH}/{dname}-{mname}-{rseed}-{shuffle_params}.npy',stacked)
    
#     with open(f'{dname}-{mname}-{rseed}.coai','wb') as w:
#         dill.dump(GRP,w)

def test_group_recalculate():
    (Xtrain,ytrain), (Xvalid,yvalid), (Xtest,ytest), costs, groups, extras = load_outpatient()
    
    Xtv = pd.concat([Xtrain,Xvalid])
    ytv = np.hstack((ytrain,yvalid))
    
    unique_costs = np.array([costs[groups==g].mean() for g in np.unique(groups)])
    dthresh = np.linspace(np.min(unique_costs),np.sum(unique_costs),11)
    
    coaimodel = knapsack.GroupOptimizer(lgb.LGBMClassifier(),OneDimExplainer)
    coaimodel.fit(Xtv,ytv,costs,groups,dthresh)
    coaimodel.score_models_proba(Xtest,ytest,roc_auc_score)
    cegbmodel = cegb.CEGBOptimizer(model=lgb.LGBMClassifier(),lambdas=np.logspace(-5,5,11))
    cegbmodel.fit(Xtv,ytv,costs)
    cegbmodel.score_models_proba(Xtest,ytest,roc_auc_score)
    
    old_coai_costs, old_coai_scores = coaimodel.model_costs, coaimodel.model_scores
    old_cegb_costs, old_cegb_scores = cegbmodel.model_costs, cegbmodel.model_scores
    
    coaimodel.recalculate_costs(costs,groups)
    cegbmodel.recalculate_costs(costs,groups)
    
    print(f'CoAI costs should be the same before and after: {(old_coai_costs==coaimodel.model_costs).mean():.2f}\% match')
    
    print(f'CEGB costs should be different before and after: {(old_cegb_costs==cegbmodel.model_costs).mean():.2f}\% match')
    return (old_coai_costs,coaimodel.model_costs,old_cegb_costs,cegbmodel.model_costs)

def test_cegb_error():
    costs, scores = train_costperf('trauma','cegb',92,None)
    
if __name__ == '__main__': main()

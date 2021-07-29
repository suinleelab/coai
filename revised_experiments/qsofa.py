import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
import sklearn
from sklearn import impute
from matplotlib import pyplot as plt
import matplotlib_venn as venn
from tqdm import tqdm as tqdm
from collections import Counter
import itertools

import config

from load_eicu import load_eicu

np.random.seed(100)

EPATH = config.EICU_DPATH

def qsofa_score(split_seed=None):
    (Xtrain,ytrain), (Xvalid,yvalid), (Xtest,ytest), costs, groups, extras = load_eicu(split_seed=split_seed)
    
    all_data = extras['all_data']
    
    Xtv = pd.concat([Xtrain,Xvalid])
    ytv = np.hstack((ytrain,yvalid))

    # Load processed BP data
    qsofa_bps = np.load(f"{EPATH}/qsofa_bp.npy",allow_pickle=True)
    qbp_map = {ptid:val for ptid,val in qsofa_bps}

    # Map to train/test inds
    qbp_train = np.array([qbp_map[ptid] for ptid in Xtv.index])
    qbp_test = np.array([qbp_map[ptid] for ptid in Xtest.index])

    # Process NaNs
    qbp_train[np.isnan(qbp_train)]=0
    qbp_test[np.isnan(qbp_test)]=0

    # qSOFA
    qtrain = Xtv[['verbal','respiratoryrate','meanbp']].copy()
    invtrain = qtrain<0
    qtrain['verbal'] = Xtv[['verbal','motor','eyes']].sum(1)<15
    qtrain['respiratoryrate'] = qtrain['respiratoryrate']>=22
    qtrain[invtrain]=0
    qtrain['meanbp'] = qbp_train
    qtrain.columns = ['gcs','respiratoryrate','meanbp']

    qtest = Xtest[['verbal','respiratoryrate','meanbp']].copy()
    invtest = qtest<0
    qtest['verbal'] = Xtest[['verbal','motor','eyes']].sum(1)<15
    qtest['respiratoryrate'] = qtest['respiratoryrate']>=22
    qtest[invtest]=0
    qtest['meanbp'] = qbp_test
    qtest.columns = ['gcs','respiratoryrate','meanbp']

    imp = impute.SimpleImputer()
    ss = sklearn.preprocessing.StandardScaler()

    qtrain_imp = imp.fit_transform(qtrain)
    qtrain_ss = ss.fit_transform(qtrain_imp)

    qtest_imp = imp.transform(qtest)
    qtest_ss = ss.transform(qtest_imp)

    qmodel = sklearn.linear_model.LogisticRegression()
    qmodel.fit(qtrain_ss,ytv)

    qvars = ['verbal','motor','eyes']+['respiratoryrate','meanbp']
    qpd = pd.DataFrame(np.zeros((1,Xtv.shape[1])),columns=Xtv.columns)
    qc = qmodel.coef_.flatten()
    qpd[qvars] = (qc[0],qc[0],qc[0],*qc[1:])

    qimps = qpd.values.flatten()
    qimps_binary = (qimps>0).astype(float)
    
    return qtest.sum(1)
# Build a PACT model
import pandas as pd
import config
import numpy as np
from sklearn.model_selection import train_test_split
from tuning import bootstrap_set
import scipy
import sklearn
import datetime
from datetime import datetime as dt, date
from tqdm import tqdm
from collections import defaultdict, Counter
from xlrd import XLRDError
from shap import LinearExplainer
from coai import knapsack
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer as Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SURVEYPATH = config.ED_SURVEYPATH

def load_cost_dict():
    with open(SURVEYPATH,'r') as f:
        header = f.readline()
        cost_dict = {line.split(',')[0]: float(line.split(',')[4]) for line in f}
    return defaultdict( lambda: config.ED_LOWCOST, cost_dict)

def pact_score(Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,costs,verbose=False):
    columns = Xtrain.columns#np.array(pd.read_pickle('../data/ed-trauma/'+"Xtrain_raw_cat.pickle").columns)
    Xtrain = pd.DataFrame(data=np.vstack((Xtrain,Xvalid)),columns=columns)
    ytrain = np.hstack((ytrain,yvalid))
    Xtest = pd.DataFrame(data=Xtest,columns=columns)
    ytest = ytest
    all_measurability = load_cost_dict()#{fname:cost for fname, cost in zip(columns,costs)}
    # PACT Score
    pact_meas = {"shock_index":all_measurability['scenefirstpulse']+all_measurability['scenefirstbloodpressure'],
                 "age":all_measurability['age'],
                 "not_mvc":all_measurability['causecode'],
                 "gcs":np.sum(all_measurability[f'scenegcs{k}'] for k in ['eye','motor','verbal']),
                 "intub":all_measurability['intub'],
                "cpr":all_measurability['cpr']}

    lr_pact = LogisticRegression()
    Xtrain_pact = pd.DataFrame()
    Xtrain_pact["shock_index"] = Xtrain['scenefirstpulse']/(Xtrain['scenefirstbloodpressure']+1)
    Xtrain_pact["age"] = Xtrain['age']
    bike_mv_mc_ped = np.array([2,14,15,18])
    Xtrain_pact["not_mvc"] = ~np.isin(Xtrain['causecode'].values,bike_mv_mc_ped)
    Xtrain_pact["gcs"] = 15-Xtrain['scenegcs']
    Xtrain_pact["cpr"] = Xtrain['cpr']
    Xtrain_pact["intub"] = Xtrain['intub']
    imp = Imputer()
    ss = StandardScaler()
    Xtrain_pact_imp = imp.fit_transform(Xtrain_pact.values.astype('float'))
    Xtrain_pact_ss = ss.fit_transform(Xtrain_pact_imp)


    Xtest_pact = pd.DataFrame()
    Xtest_pact["shock_index"] = Xtest['scenefirstpulse']/(Xtest['scenefirstbloodpressure']+1)
    Xtest_pact["age"] = Xtest['age']
    Xtest_pact["not_mvc"] = ~np.isin(Xtest['causecode'].values,bike_mv_mc_ped)
    Xtest_pact["gcs"] = 15-Xtest['scenegcs']
    Xtest_pact["cpr"] = Xtest['cpr']
    Xtest_pact["intub"] = Xtest['intub']
    Xtest_pact_imp = imp.transform(Xtest_pact.values.astype('float'))
    Xtest_pact_ss = ss.transform(Xtest_pact_imp)

    pact_lr = LogisticRegression()
    pact_lr.fit(Xtrain_pact_ss,ytrain)
    if verbose: print ("PACT ROC",roc_auc_score(ytest,pact_lr.predict_proba(Xtest_pact_ss)[:,1]))
    pact_cost = np.sum(list(pact_meas.values()))
    if verbose: print ("PACT Cost",  pact_cost)
    
    costvec = np.array([pact_meas[c] for c in Xtrain_pact.columns])
    exp = LinearExplainer
    model = LogisticRegression()
    DIO = knapsack.IncreasingCostRetainer(model,exp)
    DIO.fit(Xtrain_pact_ss,ytrain,costvec)
    DIO.score_models_proba(Xtest_pact_ss,ytest,roc_auc_score)
    
    preds = pact_lr.predict_proba(Xtest_pact_ss)[:,1]
    
    return pact_cost,roc_auc_score(ytest,preds), Xtest_pact_ss, pact_lr,DIO, Xtest_pact.columns, preds
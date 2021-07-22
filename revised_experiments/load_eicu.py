import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
import sklearn
from sklearn import impute, preprocessing, compose
from matplotlib import pyplot as plt
import matplotlib_venn as venn
from tqdm import tqdm as tqdm
from collections import Counter
import itertools

import config

# np.random.seed(100)

EPATH = config.EICU_DPATH
FIXED_SEED = config.EICU_FIXED_SEED

def load_eicu(onehot=False, split_seed=None):
    apsvar = pd.read_csv(f"{EPATH}/apacheApsVar.csv")
    results = pd.read_csv(f"{EPATH}/apachePatientResult.csv")
    predvar = pd.read_csv(f"{EPATH}/apachePredVar.csv")
    patients = pd.read_csv(f"{EPATH}/patient.csv")
    hospitals = pd.read_csv(f"{EPATH}/hospital.csv")

    # Only want apache IVa score
    results = results.iloc[results['apacheversion'].values=='IVa']
    
    # Filter to where existing scores have a value
    exist_score_cols = ['acutephysiologyscore','apachescore','predictedhospitalmortality']
    r_filter = (results[exist_score_cols]<=0).any(axis=1)
    results = results[~r_filter]

    aps_id = apsvar.set_index('patientunitstayid')
    predvar_id = predvar.set_index('patientunitstayid')
    results_id = results.set_index('patientunitstayid')
    patients_id = patients.set_index('patientunitstayid')

    all_pred = aps_id.join(predvar_id,how='outer',rsuffix='_aps')
    all_pred=all_pred.drop(columns=[c for c in all_pred.columns if '_aps' in c])
    all_data = all_pred.join(results_id,how='inner',rsuffix='_results')
    all_data = all_data.drop(columns=['apacheapsvarid','apachepredvarid','apachepatientresultsid'])

    predvar_cols = ['intubated', 'vent', 'dialysis', 'eyes', 'motor', 'verbal', 'meds',
           'urine', 'wbc', 'temperature', 'respiratoryrate', 'sodium', 'heartrate',
           'meanbp', 'ph', 'hematocrit', 'creatinine', 'albumin', 'pao2', 'pco2',
           'bun', 'glucose', 'bilirubin', 'fio2', 'gender',
           'bedcount', 'admitsource', 'graftcount', 'age', 'admitdiagnosis','ventday1',
           'oobventday1', 'oobintubday1', 'diabetes',
           'ejectfx', 'visitnumber',
           'amilocation']#, 'day1meds', 'day1verbal', 'day1motor', 'day1eyes',
           #'day1pao2', 'day1fio2']  # These vars are redundant
    predvars = all_data[predvar_cols]

    outcome = (all_data['actualhospitalmortality'].values=='EXPIRED').astype(int)
#     scores = all_data[exist_score_cols]

    pt_hospitals = patients_id.loc[predvars.index,'hospitalid'].values
    unique_hospitals = np.unique(pt_hospitals)

#     hosp_test = hospitals['hospitalid'].iloc[np.isin(hospitals['region'].values,('Northeast'))].values
    
    rand = np.random.RandomState((FIXED_SEED if split_seed is None else split_seed))
    unique_shuf = rand.permutation(unique_hospitals)
    
    train_cut, val_cut = int(0.64*len(unique_hospitals)), int(0.8*len(unique_hospitals))
    hosp_train, hosp_valid, hosp_test = unique_shuf[:train_cut], unique_shuf[train_cut:val_cut], unique_shuf[val_cut:]
#     Split based on region -- hard to do 100 random TT splits tho
#     nhosp = unique_hospitals.shape[0]
#     # hosp_train, hosp_valid, hosp_test = unique_shuf[:int(0.64*nhosp)],unique_shuf[int(0.64*nhosp):int(0.8*nhosp)],unique_shuf[int(0.8*nhosp):]
#     hosp_tv = rand.permutation(hospitals['hospitalid'].iloc[np.isin(hospitals['region'].values,('Midwest','West','South'))].values)
#     hosp_train = hosp_tv[:int(0.8*len(hosp_tv))]
#     hosp_valid = hosp_tv[int(0.8*len(hosp_tv)):]
#     hosp_test = hospitals['hospitalid'].iloc[np.isin(hospitals['region'].values,('Northeast'))].values
    train_inds = np.isin(pt_hospitals,hosp_train)
    valid_inds = np.isin(pt_hospitals,hosp_valid)
    test_inds = np.isin(pt_hospitals,hosp_test)
    
    Xtrain, ytrain = predvars.iloc[train_inds],outcome[train_inds]
    Xvalid, yvalid = predvars.iloc[valid_inds],outcome[valid_inds]
    Xtest, ytest = predvars.iloc[test_inds],outcome[test_inds]
    
#     scoretrain, scorevalid, scoretest = scores.iloc[train_inds], scores.iloc[valid_inds], scores.iloc[test_inds]
    
    for c in ['admitdiagnosis']:#, 'physicianspeciality', 'physicianinterventioncategory']:
        Xtrain[c] = Xtrain[c].astype('category')
        cats = Xtrain[c].cat.categories
        cat_type = pd.api.types.CategoricalDtype(categories=cats,ordered=False)
        Xvalid[c] = Xvalid[c].astype(cat_type)
        Xtest[c] = Xtest[c].astype(cat_type)
        
    # Encode -1s as NaNs
    for df in (Xtrain,Xvalid,Xtest):
        df.replace(to_replace=-1,value=np.nan,inplace=True)
#     if return_all: return (Xtrain,ytrain), (Xvalid,yvalid), (Xtest,ytest), np.ones(Xtrain.shape[1]), np.arange(Xtrain.shape[1]), 

    # Unit costs and per-column groups
    costs, groups = np.ones(Xtrain.shape[1]), np.arange(Xtrain.shape[1])
    if onehot:
        transformer = compose.ColumnTransformer([
            ('onehot',preprocessing.OneHotEncoder(sparse=False),['admitdiagnosis'])],
        remainder='passthrough')
        Xtrain_enc = transformer.fit_transform(Xtrain)
        Xvalid_enc = transformer.transform(Xvalid)
        Xtest_enc = transformer.transform(Xtest)
        ohcols = transformer.transformers_[0][1].get_feature_names(['admitdiagnosis'])
        newcols = list(ohcols)+[c for c in Xtrain.columns if c!='admitdiagnosis']
        Xtrain = pd.DataFrame(data=Xtrain_enc,columns=newcols,index=Xtrain.index)
        Xvalid = pd.DataFrame(data=Xvalid_enc,columns=newcols,index=Xvalid.index)
        Xtest = pd.DataFrame(data=Xtest_enc,columns=newcols,index=Xtest.index)
        ndiag = len(ohcols)
        diag_group = np.zeros(ndiag,dtype='int')
        other_group = np.arange(len(newcols)-ndiag)+1
        groups = np.hstack((diag_group,other_group))
        costs = np.ones_like(groups)
    return (Xtrain,ytrain), (Xvalid,yvalid), (Xtest,ytest), costs, groups, {'all_data':all_data}
    
def aps_baselines(split_seed=None):
    (Xtrain,ytrain), (Xvalid,yvalid), (Xtest,ytest), costs, groups, extras = load_eicu(split_seed=split_seed)
    all_data = extras['all_data']
    strain,svalid,stest = [all_data.loc[d.index,['predictedhospitalmortality','apachescore','acutephysiologyscore']] for d in (Xtrain,Xvalid,Xtest)]
    for df in strain,svalid,stest:
        df.rename(columns = {'predictedhospitalmortality':'apacheiva','apachescore':'apacheiii','acutephysiologyscore':'aps'},inplace=True)
#     apachepreds = all_data.loc[Xtest.index]['predictedhospitalmortality'].values
#     apspreds = all_data.loc[Xtest.index]['acutephysiologyscore'].values
#     apache3preds = all_data.loc[Xtest.index]['apachescore'].values
    return strain,svalid,stest#{'apacheiva': apachepreds, 'apacheiii': apache3preds, 'aps': apspreds}
import pfrl, pfrl.knapsack, pfrl.rfe, pfrl.plots, pfrl.cegb
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import shap
import itertools
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm_notebook
from matplotlib import pyplot as plt

import config

from sklearn.metrics import r2_score, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer as Imputer

DATADIR = config.OUTPATIENT_DPATH

def load_outpatient(split_seed=100):
    # Data loading
    X = pd.read_csv(DATADIR+"X_nhanes_binary.csv").drop(columns="Unnamed: 0")
    y = np.load(DATADIR+"y_nhanes_binary.npy")
    Xtrain_raw, Xtest_raw, ytrain, ytest = train_test_split(X,y,train_size=0.8,random_state=split_seed)
#     imp = Imputer()
#     ss = StandardScaler()
#     Xtrain_imp = imp.fit_transform(Xtrain_raw)
#     Xtest_imp = imp.transform(Xtest_raw)
#     Xtrain_ss = ss.fit_transform(Xtrain_imp)
#     Xtest_ss = ss.transform(Xtest_imp)
#     Xtrain_ss = pd.DataFrame(data=Xtrain_ss,columns=Xtrain_raw.columns)
#     Xtest_ss = pd.DataFrame(data=Xtest_ss,columns=Xtest_raw.columns)
    
    Xtt, Xtv, ytt, ytv = train_test_split(Xtrain_raw,ytrain,train_size=0.8,random_state=split_seed)
    
    # Costs
    feature_groups = {
    'bun': 0,
    'age': 1,
    'alkaline_phosphatase': 2,
    'band_neutrophils': 3,
    'basophils': 3,
#     'bmi': ,
    'calcium': 4,
    'cholesterol': 5,
    'creatinine': 6,
    'eosinophils': 3,
    'height': 7,
    'hematocrit': 8,
    'hemoglobin': 8,
    'lymphocytes': 3,
    'monocytes': 3,
    'physical_activity': 9,
    'platelets': 10,
    'potassium': 11,
    'pulse_pressure': 12,
    'red_blood_cells': 13,
    'sedimentation_rate': 14,
    'segmented_neutrophils': 3,
    'serum_albumin': 15,
    'serum_protein': 16,
    'sex': 17,
    'sgot': 26,
    'sodium': 18,
    'systolic_blood_pressure': 19,
    'total_bilirubin': 20,
    'uric_acid': 21,
    'urine_albumin': 22,
    'urine_glucose': 23,
    'urine_hematest': 24,
    'urine_ph': 24,
    'weight': 25,
    'white_blood_cells': 10
}
    
    groupnames = np.array(['BUN','Age','Alkaline Phosphatase','CBC w/Diff','Calcium','Cholesterol','Creatinine','Height','Hemoglobin',
             'Physical Activity','CBC Auto','Potassium','Pulse Pressure','Red Blood Cells','Sedimentation Rate','Serum Albumin',
             'Serum Protein','Sex','Sodium','Systolic BP','Total BIlirubin','Uric Acid','Urine Albumin','Urine Glucose','Urinalysis',
             'Weight','SGOT'])
    
    costfile = "feature_costs.txt"
    feature_costs = {}
    with open(DATADIR+costfile) as f:
        for line in f:
            name, cost = line.split('\t')
            try:
                feature_costs[name.lower()]=float(cost)
            except ValueError:
                feature_costs[name]=np.nan

    # Mean impute missing costs
    meancost = np.nanmean([v for v in feature_costs.values()])
    for k in feature_costs:
        if np.isnan(feature_costs[k]):
            feature_costs[k]=meancost
    
    cvec = pd.DataFrame(np.zeros_like(groupnames).reshape(-1,1),index=groupnames,columns=['Feature Group Cost (Dollars)'])
    for f in sorted(feature_groups.keys()):
        nice_name = groupnames[feature_groups[f]]
        cost = feature_costs[f]
        cvec.loc[nice_name]=f'{cost:.2f}'
#     print(cvec.to_latex())

    all_groups = {}
    for f in Xtrain_raw.columns:
        for k in feature_groups.keys():
            if k in f.lower():
                all_groups[f]=feature_groups[k]
    final_groups = np.array([all_groups[f] for f in Xtrain_raw.columns])
    
    all_costs = {}
    for f in Xtrain_raw.columns:
        for k in feature_costs.keys():
            if k in f.lower():
                all_costs[f]=feature_costs[k]
    final_costs = np.array([all_costs[f] for f in Xtrain_raw.columns])
    
    final_costs += config.OUTPATIENT_LOWCOST
    
    return (Xtt,ytt), (Xtv,ytv), (Xtest_raw,ytest), final_costs, final_groups, {}
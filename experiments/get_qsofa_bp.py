import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
import sklearn
from sklearn import impute
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
from collections import Counter, defaultdict
import itertools, os
import warnings

BP_PATH = '../data/eicu/qsofa_bp.npy'
np.random.seed(100)

def load_bp():
    warnings.warn('Loading cached BP data...')
    try: return np.load(BP_PATH,allow_pickle=True)
    except (OSError, FileNotFoundError): return generate_bp()

def loadcsv(fname):
    try:
        return pd.read_csv(fname,low_memory=True,memory_map=True)
    except (OSError ,FileNotFoundError) as e:
        raise e#("eICU data not found! Data must be acquired separately and placed in the 'data/eicu' directory.")
    
def generate_bp():
    warnings.warn('Cached BP data not found, regenerating...')
    # Load data
#     try: 
#         apsvar = pd.read_csv("../data/eicu/apacheApsVar.csv",low_memory=True,memory_map=True)
#         results = pd.read_csv("../data/eicu/apachePatientResult.csv",low_memory=True,memory_map=True)
#         patients = pd.read_csv("../data/eicu/patient.csv",low_memory=True,memory_map=True)
#         hospitals = pd.read_csv("../data/eicu/hospital.csv",low_memory=True,memory_map=True)
#         vperiodic = pd.read_csv("../data/eicu/vitalPeriodic.csv",low_memory=True,memory_map=True)
#         nursecharting = pd.read_csv('../data/eicu/nurseCharting.csv',low_memory=True,memory_map=True)
#         physical = pd.read_csv('../data/eicu/physicalExam.csv',low_memory=True,memory_map=True)
#     except (OSError, FileNotFoundError) as e:
#         raise e("eICU data not found! Data must be acquired separately and placed in the 'data/eicu' directory.")
        
        

    # Filter to rows with BP
#     print('loading physical...')
    physical = loadcsv('../data/eicu/physicalExam.csv')
    bp_phys_inds = np.char.find(np.char.lower(physical['physicalexampath'].values.astype('unicode')),'vital sign and physiological data/bp')>0
    bp_physical = physical.iloc[bp_phys_inds].copy()
    del physical

    # Filter to rows with BP
#     print('loading nursing...')
    nursecharting = loadcsv('../data/eicu/nurseCharting.csv')
    bp_nurse_inds = np.char.find(nursecharting['nursingchartcelltypevalname'].values.astype('unicode'),'BP')>0
    bp_nursecharting = nursecharting.iloc[bp_nurse_inds].copy()
    del nursecharting

    # Group real-time vitals by patient ID
#     print('loading periodic...')
    vitals = defaultdict(list)
    with open('../data/eicu/vitalPeriodic.csv','r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            row = line.strip().split(',')
            ptid,offset,syst,mean,diast = [(row[i]) for i in [1,2,9,10,11]]
            syst, mean, diast = [float('nan') if i=='' else int(i) for i in [syst,mean,diast]]
            if not np.isnan(syst) or not np.isnan(mean) or not np.isnan(diast):
                vitals[ptid].append((offset,syst,mean,diast))
#     vperiodic = loadcsv('../data/eicu/vitalPeriodic.csv')
#     for ptid, offset, syst, mean, diast in (vperiodic[['patientunitstayid','observationoffset','systemicsystolic','systemicmean','systemicdiastolic']].values):
#         vitals[ptid].append((offset,syst,mean,diast))
#     del vperiodic

    # Physical exam vitals
#     print('processing...')
    physvitals = {k: defaultdict(list) for k in bp_physical['physicalexamvalue'].unique()}
    for ptid, offset, vtype, val in (bp_physical[['patientunitstayid','physicalexamoffset','physicalexamvalue','physicalexamtext']].values):
        physvitals[vtype][ptid].append((offset,val))

    # Nursing vitals
    nursevitals = {k: defaultdict(list) for k in bp_nursecharting['nursingchartcelltypevalname'].unique()}
    for ptid, offset, vtype, val in (bp_nursecharting[['patientunitstayid','nursingchartoffset','nursingchartcelltypevalname','nursingchartvalue']].values):
        nursevitals[vtype][ptid].append((offset,val))

    # Consolidate all vitals
    all_vitals = {k: defaultdict(list) for k in ['systolic','diastolic','mean']}
    for vdict in [nursevitals,physvitals]:
        for k in vdict:
            fkey = 'systolic' if 'systolic' in k.lower() else 'diastolic' if 'diastolic' in k.lower() else 'mean' if 'mean' in k.lower() else 'oops'
            for ptid in (vdict[k]):
                all_vitals[fkey][ptid].extend(vdict[k][ptid])
    for ptid in (vitals):
        for offset, systolic, mean, diastolic in vitals[ptid]:
            for k,v in [('systolic',systolic),('mean',mean),('diastolic',diastolic)]:
                all_vitals[k][ptid].append((offset,v))

    # Get qSOFA values
    apsvar = loadcsv('../data/eicu/apacheApsVar.csv')
    qsofa_syst = np.zeros(apsvar.shape[0])
    qsofa_syst *= np.nan
    for i, ptid in (enumerate(apsvar['patientunitstayid'].values)):
        if len(all_vitals['systolic'][ptid])==0: continue
        systolics = np.vstack(all_vitals['systolic'][ptid]).astype(float)
        hr24 = systolics[:,0]<=(24*60)
        nonnan = ~np.isnan(systolics[:,1])
        vals = systolics[:,1][hr24&nonnan]
        qsofa_syst[i] = (vals<=100).any()

    # Stack with patient IDs
    qsofa_data = np.vstack((apsvar['patientunitstayid'],qsofa_syst)).astype(object)
    qsofa_data[0] = qsofa_data[0].astype(int)
    qsofa_data = qsofa_data.T

    # Cache
    np.save(BP_PATH,qsofa_data)
    return qsofa_data
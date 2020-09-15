from sklearn import base, dummy
import numpy as np
import pandas as pd
import os, subprocess, shutil
import dill
import tempfile
import lightgbm as lgb
import shap
from scipy.special import softmax
from .utils import iterator
from .base import CostAwareOptimizer

DEFAULT_PARAMS = {
    'dataset': 'coai',
    'classes': 2,
    'features': None,
    'nn_size': 128,
    'difficulty': 1000
}

def get_preds_labels(df):
    y,c,d = df['y'].values, df['correct'].values, df['done'].values
#     stop_idx = np.where(y<0)[0][0]
    valid_inds = d==1
    yv, cv = y[valid_inds], c[valid_inds].astype(bool)
    pv = yv.copy()
    pv[~cv] = 1-pv[~cv]
    return yv,pv

def get_probs(outputs,qs):
    done = outputs['done'].values==1
    return softmax(qs[done,:2],axis=1)

class CWCFClassifier(object):
    def __init__(self,lmbd=1.0,name='coai',costs=None,**kwargs):
        self.lmbd = lmbd
        self.costs = costs
        self.name = name
        self.params = DEFAULT_PARAMS.copy()
        for k,v in kwargs.items():
            self.params[k]=v
        self.tmpdir = tempfile.TemporaryDirectory()
        shutil.copytree('../cwcf_fixed',self.tmpdir.name+'/cwcf')
            
    def fit(self,Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest):
        # Default feature cost is 1
        if self.costs is None: self.costs = np.ones(Xtrain.shape[1])
#         for df in [Xtrain,Xvalid,Xtest]:
            
        
        # Format train/val/test data for CWCF
        train_matrix = pd.DataFrame(np.hstack([Xtrain.astype(float),ytrain.reshape(-1,1).astype(int)]),
                                   columns=list(range(Xtrain.shape[1]))+['_label'],
                                   index = np.arange(Xtrain.shape[0]))
        valid_matrix = pd.DataFrame(np.hstack([Xvalid.astype(float),yvalid.reshape(-1,1).astype(int)]),
                                   columns=list(range(Xvalid.shape[1]))+['_label'],
                                   index = np.arange(Xvalid.shape[0]))
        test_matrix = pd.DataFrame(np.hstack([Xtest.astype(float),ytest.reshape(-1,1).astype(int)]),
                                   columns=list(range(Xtest.shape[1]))+['_label'],
                                   index = np.arange(Xtest.shape[0]))
        
        # Metadata, ie feature means, stdevs, costs
        meta = pd.DataFrame(np.zeros((Xtrain.shape[1],3)), columns=['avg','std','cost'])
        meta.iloc[:,0] = Xtrain.mean(axis=0).values
        meta.iloc[:,1] = Xtrain.std(axis=0).values
        meta.iloc[:,2] = self.costs
        
        # HPC files
        hpc = pd.DataFrame(np.zeros((np.max([Xtrain.shape[0],Xvalid.shape[0],Xtest.shape[0]]),3)),
                           columns=['train','validation','test'])
        
        # Parameters to go in config file
        self.params['features']=Xtrain.shape[1]
        with open(f'{self.tmpdir.name}/cwcf/config_datasets/{self.name}.py','w') as w:
            for k,v in self.params.items():
                if type(v) is str: w.write(f'{k.upper()} = "{v}"\n')
                else: w.write(f'{k.upper()} = {v}\n')
        
        # Save
        train_matrix.to_pickle(f'{self.tmpdir.name}/cwcf/data/{self.name}-train')
        valid_matrix.to_pickle(f'{self.tmpdir.name}/cwcf/data/{self.name}-val')
        test_matrix.to_pickle(f'{self.tmpdir.name}/cwcf/data/{self.name}-test')
        meta.to_pickle(f'{self.tmpdir.name}/cwcf/data/{self.name}-meta')
        hpc.to_pickle(f'{self.tmpdir.name}/cwcf/data/{self.name}-hpc')
        
        # Link data directory to the place cwcf will look for it
        os.symlink(f'{self.tmpdir.name}/cwcf/data',f'{self.tmpdir.name}/data')
        
        # Run CWCF
        result = subprocess.run(['python3','main.py','--dataset',self.name,'--flambda',str(self.lmbd),'--use_hpc','0','--pretrain','1'],cwd=self.tmpdir.name+'/cwcf')
        
        # Compile results in result_dir
        result_dir = f'{self.tmpdir.name}/cwcf/{self.name}-{self.lmbd}'
        os.mkdir(result_dir)
        move_files = [f for f in os.listdir(self.tmpdir.name+'/cwcf') if f.startswith('run') or f.startswith('model')]
        for file in move_files:
            shutil.move(f'{self.tmpdir.name}/cwcf/{file}',result_dir)
        results = {}
        for dset in ['trn','val','tst']:
            fname = f'{result_dir}/run_{dset}_best_allpreds.csv'
            model_outputs = pd.read_csv(fname)
            qname = f'{result_dir}/run_{dset}_best_allqs.csv'
            model_q = np.genfromtxt(qname,delimiter=',')
            probs = get_probs(model_outputs,model_q)
            labels, preds = get_preds_labels(model_outputs)#model_outputs['y'].values, model_outputs['correct'].values)
            fcost = f'{result_dir}/run_{dset}_best_perf.dat'
            with open(fcost) as f: info = {k:float(v) for k,v in zip(['reward','len','cost','hpc','accuracy'],f.read().split())}
            results[dset] = {'preds': preds, 'labels': labels, 'probs': probs, **info}#(preds,labels,info)
            
        
        # Done
        return results    
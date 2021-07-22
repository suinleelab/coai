from sklearn import base, dummy, metrics
import numpy as np
import pandas as pd
import os, subprocess, shutil, itertools
import dill
import tempfile
import lightgbm as lgb
import shap
import multiprocessing as mp
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

CWCF_FIXED_PATH = '/'.join(__file__.split('/')[:-2])

def get_preds_labels(df):
    y,c,d = df['y'].values, df['correct'].values, df['done'].values
#     stop_idx = np.where(y<0)[0][0]
    valid_inds = d==1
    yv, cv = y[valid_inds], c[valid_inds].astype(bool)
    pv = yv.copy()
    pv[~cv] = 1-pv[~cv]
    return yv,pv

def get_probs(outputs,qs,n_classes=2):
    done = outputs['done'].values==1
    return softmax(qs[done,:n_classes],axis=1)

def single_job(tk):
    t, kwargs = tk
    M = CWCFClassifier(lmbd=t[0],costs=t[1],difficulty=t[2],classes=t[3],gpus=t[4],dirname=t[5],**kwargs)
    results = M.fit(t[6],t[7],t[8],t[9],t[10],t[11])
    return results

def get_cwcf(Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,costs,lmbds,gpus=list(range(8)),njobs=32,dirname=None,metric=metrics.roc_auc_score,**kwargs):
        GRP = lambda x: x
        mcosts,mscores = [],[]
        difficulty = kwargs.pop('difficulty') if 'difficulty' in kwargs else 1000
        nclass = kwargs.pop('classes') if 'classes' in kwargs else 2
        tlist = [((lmbd,costs,difficulty,nclass,(gpu,),dirname,Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest),
                 kwargs) 
                 for i,(lmbd,gpu) in enumerate(zip(lmbds,itertools.cycle(gpus)))]
        
        if njobs>1:
            P = mp.Pool(njobs)
            results = P.map(single_job,tlist)
        else: results = map(single_job,tlist)
        for r in results:
            if 'tst' not in r:
                mcosts.append(np.nan),mscores.append(np.nan)
            else:
                mcosts.append(r['tst']['cost'])
                mscores.append(metric(r['tst']['labels'],r['tst']['probs'][:,1]))
        GRP.model_costs, GRP.model_scores = np.array(mcosts), np.array(mscores)
        return GRP


class CWCFClassifier(object):
    def __init__(self,lmbd=1.0,name='coai',gpus=None,costs=None,groups=None,dirname=None,lagrange=False,**kwargs):
        self.lmbd = lmbd
        self.costs = costs
        self.groups = groups
        self.name = name
        self.lagrange = lagrange
        self.params = DEFAULT_PARAMS.copy()
        for k,v in kwargs.items():
            self.params[k]=v
        self.tmpdir = tempfile.TemporaryDirectory(dir=dirname)
        self.gpus = gpus
        shutil.copytree(f'{CWCF_FIXED_PATH}/cwcf_{"lagrange" if self.lagrange else "fixed"}',self.tmpdir.name+'/cwcf')
            
    def fit(self,Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest):
        # Default feature cost is 1
        if self.costs is None: self.costs = np.ones(Xtrain.shape[1])
        if self.groups is None: self.groups = np.arange(Xtrain.shape[1])
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
        meta = pd.DataFrame(np.zeros((Xtrain.shape[1],4)), columns=['avg','std','cost','group'])
        meta.iloc[:,0] = Xtrain.mean(axis=0).values
        meta.iloc[:,1] = Xtrain.std(axis=0).values
        meta.iloc[:,2] = self.costs
        meta.iloc[:,3] = self.groups
        
        # HPC files
        hpc = pd.DataFrame(np.zeros((np.max([Xtrain.shape[0],Xvalid.shape[0],Xtest.shape[0]]),3)),
                           columns=['train','validation','test'])
        
        # Parameters to go in config file
        self.params['features']=Xtrain.shape[1]
        self.params['groups']=np.unique(self.groups).shape[0]
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
        print([x.shape for x in [train_matrix,valid_matrix,test_matrix,meta,hpc]])
        
        # Link data directory to the place cwcf will look for it
        os.symlink(f'{self.tmpdir.name}/cwcf/data',f'{self.tmpdir.name}/data')
        
        # Set up environment
        env = {} if self.gpus is None else {'CUDA_DEVICE_ORDER':'PCI_BUS_ID',
                   'CUDA_VISIBLE_DEVICES':','.join([str(x) for x in self.gpus])}
        
        if self.lagrange:
            args = ['python3','main.py','-use_hpc','0','-pretrain','1','-target_type','cost',self.name,str(self.lmbd)]
        else:
            # Set up non-Lagrange args
            args = ['python3','main.py','--dataset',self.name,'--flambda',str(self.lmbd),'--use_hpc','0','--pretrain','1']

        # # Set up Lagrange args
        # extras = ['-target_type','cost'] if self.lagrange else []

        # Run CWCF
        try: result = subprocess.run(args,cwd=self.tmpdir.name+'/cwcf',stdout=subprocess.PIPE,env=env)
        except:
            print(['python3','main.py','--dataset',self.name,'--flambda',str(self.lmbd),'--use_hpc','0','--pretrain','1']); exit()
        if result.returncode!=0: return {'process_results':result}
        
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
            probs = get_probs(model_outputs,model_q,n_classes=self.params['classes'] if 'classes' in self.params else 2)
            labels, preds = get_preds_labels(model_outputs)#model_outputs['y'].values, model_outputs['correct'].values)
            fcost = f'{result_dir}/run_{dset}_best_perf.dat'
            with open(fcost) as f: info = {k:float(v) for k,v in zip(['reward','len','cost','hpc','accuracy'],f.read().split())}
            results[dset] = {'preds': preds, 'labels': labels, 'probs': probs,
#                              'all_outputs':model_outputs,'all_qs':model_q,
                             **info}#(preds,labels,info)
        results['process_results'] = result
            
        
        # Done
        return results    
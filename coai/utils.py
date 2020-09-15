#! /usr/bin/python
import xgboost as xgb
import numpy as np
import re
import multiprocessing
from tqdm import tqdm
iterator = tqdm

# Calculates measurability of a tree's predictions
# ONLY WORKS FOR SINGLE TREES
class MeasureTree(object):
    def __init__(self,tree_model, feature_costs):
        assert type(tree_model)==xgb.core.Booster, "tree_model must be an XGBoost Booster!" 
        self.tree_model = tree_model
        self.tree_dump = tree_model.get_dump()[0]
        assert type(feature_costs)==dict, "feature_costs must be a dictionary!" 
        self.feature_costs = feature_costs
        self.path_cache = {}

    # Return path to a given leaf in the tree
    def path(self,leafnum):
        if leafnum in self.path_cache: return self.path_cache[leafnum]
        lines = self.tree_dump.split('\n')[::-1]
        leaf_inds = [re.search(f'^\t*{leafnum}\:',line) for line in lines]
        start = np.where(leaf_inds)[0][0]
        tabs = len(lines[start])-len(lines[start].lstrip('\t'))
        ltext = lines[start].strip()
        path = []
        for line in lines[start:]:
            t2 = len(line)-len(line.lstrip('\t'))
            if t2<tabs:
                ltext = line.strip()
                path.append(ltext[ltext.index("[")+1:ltext.index("<")])
                tabs = t2
        self.path_cache[leafnum] = (start, path)
        return start, path

    # Measurability cost for predicting on an array
    def measure_predictions(self,data):
        leaves = self.tree_model.predict(xgb.DMatrix(data),pred_leaf=True)[:,0]
        meas = []
        for leaf in leaves:
            _,p = self.path(leaf)
            meas.append(np.sum([self.feature_costs[f] for f in set(p)]))
        return np.array(meas)

# Calculates measurability of a linear model's predictions
# Assumes model has some coefficient vector, under .coef or .coef_
class MeasureLinear(object):
    def __init__(self,lin_model, feature_costs, feature_names=None):
        assert 'coef' in dir(lin_model) or 'coef_' in dir(lin_model), "lin_model must have a .coef or .coef_ attribute"
        self.lin_model = lin_model
        self.lin_coefs = lin_model.coef_.flatten() if 'coef_' in dir(lin_model) else lin_model.coef.flatten()
        assert type(feature_costs)==dict, "feature_costs must be a dictionary!"
        self.feature_costs = feature_costs
        if feature_names is None:
            feature_names = np.arange(self.lin_coefs.shape[0])
        self.pred_cost = np.sum([self.feature_costs[feature_names[i]] for i,b in enumerate(self.lin_coefs) if b!=0])

    # Measurability cost for predicting on an array
    def measure_predictions(self,data):
        ret = np.zeros(data.shape[0])
        ret[:]=self.pred_cost
        return ret
        
class SimpleTreeExplainer(object):
    def __init__(self,model,data):
        self.booster = model.get_booster()
    def shap_values(self,data):
        shaps = self.booster.predict(xgb.DMatrix(data),pred_contribs=True)
        if shaps.ndim>2:
            return np.mean(np.abs(shaps[:,:,:-1]),axis=0)
        else:
            return shaps[:,:-1]

# Pmap with a progress bar
def pmap_progress(func,iterable,nprocs=8,qsize=100,iter_len=None):
    if iter_len is None: iter_len = len(iterable)
    P = multiprocessing.Pool(nprocs)
    #P._taskqueue.qsize=qsize
    #P._taskqueue._sem = multiprocessing.BoundedSemaphore(qsize)

    results = []
    for i,t in iterator(enumerate(P.imap(func,iterable)),total=iter_len):
        results.append(t)
    return results

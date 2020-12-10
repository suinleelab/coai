#! /usr/bin/python
import numpy as np
import re
import multiprocessing
from tqdm import tqdm
iterator = tqdm
        
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

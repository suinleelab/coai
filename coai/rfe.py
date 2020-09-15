#! /usr/bin/python

import shap
import pdb
import numpy as np
from .base import CostAwareOptimizer
from tqdm import tqdm
from sklearn import model_selection
from ortools.algorithms import pywrapknapsack_solver

iterator = tqdm
    
class RFEOptimizer(CostAwareOptimizer):
    def intermediate_cost_models(self,X,y,add_to_model=False):
        tfeats, tcosts, tmodels = [], [], []
        features = list(np.arange(X.shape[1],dtype='int'))
        for i in iterator(range(self.feature_costs.shape[0])):
            curX = self.selectfeats(X,features)
            cury = y
            tfeats.insert(0,features)
            tcosts.insert(0,np.sum(self.feature_costs[features]))
            if i==0 and self.full_model is not None:
                tmodels.insert(0,self.full_model)
            else:
                tmodels.insert(0,self.newmodel())
                tmodels[0].fit(curX,cury)
            if i!=self.feature_costs.shape[0]:
                exp = self.base_explainer(tmodels[0],curX)
                shaps = exp.shap_values(curX)
                imps = (np.sum(np.abs(shaps),axis=0))/self.feature_costs[features]
                worst = np.arange(X.shape[1],dtype='int')[features][np.argmin(imps)]
                features = [v for v in features if v!=worst]
        if add_to_model:
            self.models.extend(tmodels[:-1])
            self.model_costs.extend(tcosts[:-1])
            self.model_features.extend(tfeats[:-1])
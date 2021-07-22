#! /usr/bin/python
from sklearn import base, dummy
import numpy as np
import pandas as pd
import dill
import lightgbm as lgb
import shap
from .utils import iterator
from .base import CostAwareOptimizer, OneDimExplainer

defaultmodel = lambda: lgb.LGBMClassifier()
defaultexplainer = OneDimExplainer

class CEGBOptimizer(CostAwareOptimizer):
    def __init__(self,model=None,lambdas=None):
        if model is None: model = defaultmodel()
        super().__init__(model,defaultexplainer)
        if lambdas is None: self.lambdas = np.hstack(([0.0],np.logspace(-5,5,11)))
        else: self.lambdas = lambdas

    def selectfeats(self,arr,feats):
        return arr

    def newmodel(self,newparams=None):
        if newparams is None: return super().newmodel()
        params = self.base_model.get_params()
        params.update(newparams)
        return lgb.LGBMClassifier(**params)

    def intermediate_cost_models(self,X,y,add_to_model=False):
        tfeats, tcosts, tmodels = [], [], []
        for lmbd in iterator(self.lambdas[::-1]):
            newparams = {'cegb_penalty_feature_coupled':self.feature_costs,'cegb_tradeoff':lmbd,'cegb_feature_groups':self.groups}
            model = self.newmodel(newparams)
            self.fitmodel(model,X,y)
            tmodels.append(model)
            tfeats.append(np.where(model.feature_importances_!=0)[0])
            tcosts.append(np.sum(self.feature_costs[tfeats[-1]]))
        if add_to_model:
            self.models.extend(tmodels)
            self.model_costs.extend(tcosts)
            self.model_features.extend(tfeats)
    def fit(self,X,y,costs,groups=None):
        if groups is None: groups = np.arange(costs.shape[0])
        self.groups = groups
        super().fit(X,y,costs)
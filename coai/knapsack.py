#! /usr/bin/python

import shap, pdb
from .base import CostAwareOptimizer
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import model_selection
from ortools.algorithms import pywrapknapsack_solver

iterator = tqdm

def knapsolve(values,weights,budget):
    assert values.dtype==weights.dtype==int
    values = list(values)
    weights = [list(weights)]
    solver = pywrapknapsack_solver.KnapsackSolver(pywrapknapsack_solver.KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,'globalpfrl')
    solver.Init(values,weights,[budget])
    total_importances = solver.Solve()
    best_items = [x for x in range(len(weights[0]))
                  if solver.BestSolutionContains(x)]
    best_values = [weights[0][i] for i in best_items]
    return best_items

def zerofeats(arr,feats):
        a2 = arr.copy()
        zerofeats = [i for i in range(a2.shape[1]) if i not in feats]
        if type(arr) is pd.core.frame.DataFrame: a2.iloc[:,zerofeats]=0
        else: a2[:,zerofeats]=0
        return a2

class DynamicOptimizer(CostAwareOptimizer):
    def __init__(self, model, explainer, scale_ints=None):
        super().__init__(model,explainer)
        if scale_ints is None: scale_ints = 100
        self.scale_ints = scale_ints
        self.thresholds = None
        self.global_importances = None

    def rescale_ints(self,arr):
        return (arr*self.scale_ints).astype(int)

    def feats_at_thresh(self,imps,thresh):
        return knapsolve(
                self.rescale_ints(imps),
                self.rescale_ints(self.feature_costs),
                thresh*self.scale_ints)

    def feats_costs(self,feats):
        return np.sum(self.feature_costs[feats])

    def intermediate_cost_models(self,X,y,add_to_model=False):
        explainer = self.base_explainer(self.full_model,X)
        shap_values = explainer.shap_values(X)
        global_importances = np.sum(np.abs(shap_values),axis=0)
        self.global_importances = global_importances
        tfeats, tcosts, tmodels = [], [], []
        for t in iterator(self.thresholds):
            opt_feats = self.feats_at_thresh(global_importances,t)
            opt_cost = self.feats_costs(opt_feats)
            if len(opt_feats)==0 or not self.cost_criterion(opt_cost): continue
            tfeats.append(opt_feats)
            tcosts.append(opt_cost)
            tmodels.append(self.newmodel())
            self.fitmodel(tmodels[-1],self.selectfeats(X,tfeats[-1]),y)
        if add_to_model:
            self.models.extend(tmodels)
            self.model_costs.extend(tcosts)
            self.model_features.extend(tfeats)
            
    def fit(self,X,y,feature_costs=None,thresholds=None,cost_criterion=None,**kwargs):
        if feature_costs is None:
            feature_costs = np.ones(X.shape[1])
        self.feature_costs = np.array(feature_costs)
        if thresholds is None:
            threshstep = np.min(self.feature_costs)
            threshmax = np.sum(self.feature_costs)
            thresholds = np.arange(0,threshmax,threshstep)
        self.thresholds = thresholds
        super().fit(X,y,feature_costs=feature_costs,**kwargs)           

class GroupOptimizer(DynamicOptimizer):
    def fit(self,X,y,feature_costs=None,feature_groups=None,thresholds=None,cost_criterion=None,**kwargs):
        if feature_groups is None:
            feature_groups = np.arange(X.shape[1])
        self.feature_groups = feature_groups
        self.unique_groups = np.unique(feature_groups)
        super().fit(X,y,feature_costs=feature_costs,thresholds=thresholds,cost_criterion=cost_criterion,**kwargs)
        
    def feats_at_thresh(self,imps,thresh):
        groups = self.feature_groups
        unique_groups = self.unique_groups
        costs = self.feature_costs

        grouped_imps = np.array([np.sum(imps[groups==u]) for u in unique_groups])
        grouped_costs = np.array([np.mean(costs[groups==u]) for u in unique_groups])

        opt_group_inds = knapsolve(
                self.rescale_ints(grouped_imps),
                self.rescale_ints(grouped_costs),
                thresh*self.scale_ints)
        opt_groups = np.array([unique_groups[i] for i in opt_group_inds])
        
        opt_feats = np.array([i for i in np.arange(self.feature_groups.shape[0]) if groups[i] in opt_groups],dtype=int)
        return opt_feats

    def feats_costs(self,feats):
        groups = self.feature_groups
        unique_groups = self.unique_groups
        costs = self.feature_costs
        grouped_costs = {u:np.mean(self.feature_costs[groups==u]) for u in unique_groups}

        groups_present = np.unique(groups[feats])
        return np.sum([grouped_costs[g] for g in groups_present])

    def full_cost_model(self,X,y,add_to_model=False):
        model, cost, features = super().full_cost_model(X,y,add_to_model=False)
        newcost = self.feats_costs(features)
        return (model, newcost, features)



        
            
class GreedyOptimizer(CostAwareOptimizer):
    def intermediate_cost_models(self,X,y,add_to_model=False):
        explainer = self.base_explainer(self.full_model,X)
        shap_values = explainer.shap_values(X)
        global_importances = np.sum(np.abs(shap_values),axis=0)
        gains = global_importances/self.feature_costs
        order = np.argsort(gains)[::-1]
        tfeats, tcosts, tmodels = [], [], []
        for i in iterator(range(1,len(order)+1)):
            newfeats = sorted(order[:i])
            newcost = np.sum(self.feature_costs[newfeats])
            if len(newfeats)==0 or not self.cost_criterion(newcost): continue
            tfeats.append(newfeats)
            tcosts.append(newcost)
            tmodels.append(self.newmodel())
            self.fitmodel(tmodels[-1],self.selectfeats(X,tfeats[-1]),y)
        if add_to_model:
            self.models.extend(tmodels[:-1])
            self.model_costs.extend(tcosts[:-1])
            self.model_features.extend(tfeats[:-1])
            
class IncreasingCostOptimizer(CostAwareOptimizer):
    def intermediate_cost_models(self,X,y,add_to_model=False):
        order = np.argsort(self.feature_costs)
        tfeats, tcosts, tmodels = [], [], []
        for i in iterator(range(1,len(order)+1)):
            newfeats = sorted(order[:i])
            newcost = np.sum(self.feature_costs[newfeats])
            if len(newfeats)==0 or not self.cost_criterion(newcost): continue
            tfeats.append(newfeats)
            tcosts.append(newcost)
            tmodels.append(self.newmodel())
            self.fitmodel(tmodels[-1],self.selectfeats(X,tfeats[-1]),y)
        if add_to_model:
            self.models.extend(tmodels[:-1])
            self.model_costs.extend(tcosts[:-1])
            self.model_features.extend(tfeats[:-1])
            
class DecreasingImportanceOptimizer(CostAwareOptimizer):
    def intermediate_cost_models(self,X,y,add_to_model=False):
        explainer = self.base_explainer(self.full_model,X)
        shap_values = explainer.shap_values(X)
        global_importances = np.sum(np.abs(shap_values),axis=0)
        gains = global_importances
        order = np.argsort(gains)[::-1]
        tfeats, tcosts, tmodels = [], [], []
        for i in iterator(range(1,len(order)+1)):
            newfeats = sorted(order[:i])
            newcost = np.sum(self.feature_costs[newfeats])
            if len(newfeats)==0 or not self.cost_criterion(newcost): continue
            tfeats.append(newfeats)
            tcosts.append(newcost)
            tmodels.append(self.newmodel())
            self.fitmodel(tmodels[-1],self.selectfeats(X,tfeats[-1]),y)
        if add_to_model:
            self.models.extend(tmodels[:-1])
            self.model_costs.extend(tcosts[:-1])
            self.model_features.extend(tfeats[:-1])

class DynamicRetainer(DynamicOptimizer):
    def selectfeats(self,arr,feats):
        return zerofeats(arr,feats)
    
class GreedyRetainer(GreedyOptimizer):
    def selectfeats(self,arr,feats):
        return zerofeats(arr,feats)

class IncreasingCostRetainer(IncreasingCostOptimizer):
    def selectfeats(self,arr,feats):
        return zerofeats(arr,feats)

class DecreasingImportanceRetainer(DecreasingImportanceOptimizer):
    def selectfeats(self,arr,feats):
        return zerofeats(arr,feats)

class FixedModelExactRetainer(DynamicOptimizer):
    def selectfeats(self,arr,feats):
        return zerofeats(arr,feats)

    def full_cost_model(self,X,y,add_to_model=False):
        return self.base_model, np.sum(self.feature_costs), np.arange(X.shape[1])

    def intermediate_cost_models(self,X,y,add_to_model=False):
        explainer = self.base_explainer(self.full_model,X)
        shap_values = explainer.shap_values(X)
        global_importances = np.sum(np.abs(shap_values),axis=0)
        tfeats, tcosts, tmodels = [], [], []
        for t in iterator(self.thresholds):
            opt_feats = knapsolve(
                self.rescale_ints(global_importances),
                self.rescale_ints(self.feature_costs),
                t*self.scale_ints)
            tfeats.append(opt_feats)
            tcosts.append(np.sum(self.feature_costs[tfeats[-1]]))
            tmodels.append(self.base_model)
            # self.fitmodel(tmodels[-1],self.selectfeats(X,tfeats[-1]),y)
        if add_to_model:
            self.models.extend(tmodels[:-1])
            self.model_costs.extend(tcosts[:-1])
            self.model_features.extend(tfeats[:-1])

class FixedDualModelExact(FixedModelExactRetainer):
    def __init__(self,expmodel,predmodel,explainer,scale_ints=None):
        super().__init__(expmodel,explainer,scale_ints)
        self.pred_model = predmodel
    def fit(self,X,y,feature_costs=None,thresholds=None,cost_criterion=None):
        super().fit(X,y,feature_costs=feature_costs,thresholds=thresholds,cost_criterion=cost_criterion)
        self.models = [self.pred_model]*len(self.models)
            
class FixedModelGreedyRetainer(CostAwareOptimizer):
    # def __init__(self, model, explainer, impute=False)`:
    #     super().__init__(model, explainer)
    #     self.impute = impute`

    def selectfeats(self,arr,feats):
        return zerofeats(arr,feats)

    def full_cost_model(self,X,y,add_to_model=False):
        return self.base_model, np.sum(self.feature_costs), np.arange(X.shape[1])

    def intermediate_cost_models(self,X,y,add_to_model=False):
        explainer = self.base_explainer(self.full_model,X)
        shap_values = explainer.shap_values(X)
        global_importances = np.sum(np.abs(shap_values),axis=0)
        gains = global_importances/self.feature_costs
        order = np.argsort(gains)[::-1]
        tfeats, tcosts, tmodels = [], [], []
        for i in iterator(range(1,len(order)+1)):
            newfeats = sorted(order[:i])
            newcost = np.sum(self.feature_costs[newfeats])
            if not self.cost_criterion(newcost): continue
            tfeats.append(newfeats)
            tcosts.append(newcost)
            tmodels.append(self.base_model)
            # self.fitmodel(tmodels[-1],self.selectfeats(X,tfeats[-1]),y)
        if add_to_model:
            self.models.extend(tmodels[:-1])
            self.model_costs.extend(tcosts[:-1])
            self.model_features.extend(tfeats[:-1])
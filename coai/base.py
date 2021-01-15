#! /usr/bin/python
from sklearn import base, dummy, utils
import shap
import numpy as np
import pandas as pd
import dill
import xgboost as xgb
from .utils import iterator, pmap_progress

def load_model(fname):
    with open(fname, 'rb') as f:
        return dill.load(f)
    
class OneDimExplainer(shap.TreeExplainer):
    def __init__(self,model,data):
        return super().__init__(model)
    def shap_values(self,data):
        shaps = super().shap_values(data)
        if type(shaps)==list:
            shaps = shaps[-1]
        return shaps

class CostAwareModel():
    def __init__(self,base_model,base_explainer=shap.SamplingExplainer,max_cost=np.inf,explainer_params={},global_importances=None):
        self.base_model = base_model
        self.base_explainer = base_explainer
        self.explainer_params = explainer_params
        self.global_importances = global_importances
        self.max_cost = max_cost
        # self.full_model = None
        # self.fit_kwargs_ = None

    # Return a clone of the base model
    def newmodel(self):
        return base.clone(self.base_model)

    # Subsample features of a datamatrix
    def selectfeats(self,arr,feats):
        if type(arr) is pd.core.frame.DataFrame:
            return arr.iloc[:,feats]
        else: return arr[:,feats]

    def get_global_importances(self,X):
        raise NotImplementedError

    def feats_at_thresh(self,imps,thresh):
        raise NotImplementedError

    # Fit!
    def fit(self,X,y,feature_costs=None,**kwargs):
        self.costs_ = feature_costs
        self.fit_kwargs_ = kwargs

        # Fit full cost model if needed
        if utils.validation.check_is_fitted(self.model):
            self.full_model_ = self.base_model_
        else:
            self.full_model_ = self.newmodel().fit(X,y,**kwargs)

        # Get importances
        if self.global_importances:
            self.global_importances_ = self.global_importances
        else:
            self.explainer_ = self.base_explainer(self.full_model_,X,**self.explainer_params)
            self.global_importances_ = self.get_global_importances(X)

        # Get features at cost
        self.atcost_features_ = self.feats_at_thresh(self.global_importances_,self.max_cost)
        
        # Fit at-cost model
        subX = self.selectfeats(X,self.atcost_features_)
        self.atcost_model_ = self.newmodel().fit(subX,y,**kwargs)
        self.actual_cost_ = np.sum(self.costs_[self.atcost_features_])

        # Done
        return self

    # Predict using best possible model for a given cost
    def predict(self,X):
        return self.atcost_model_.predict(self.selectfeats(X,self.atcost_features_))
    
    # Predict_proba using best possible model for a given cost
    def predict_proba(self,X):
        return self.atcost_model_.predict_proba(self.selectfeats(X,self.atcost_features_))

        
        

class CostAwareOptimizer(object):
    def __init__(self, model, explainer):
        self.base_model = model
        self.base_explainer = explainer
        self.full_model = None
        self.models = []
        self.model_costs = []
        self.model_features = []
        self.model_scores = None
        self.X = None
        self.y = None
        self.fit_kwargs = None

    # Return a clone of the base model
    def newmodel(self):
        return base.clone(self.base_model)

    # Subsample features of a datamatrix
    def selectfeats(self,arr,feats):
        if type(arr) is pd.core.frame.DataFrame:
            return arr.iloc[:,feats]
        else: return arr[:,feats]

    # Fit a model to passed data.
    def fitmodel(self,model,X,y):
        model.fit(X,y,**self.fit_kwargs)

    # Use dummy model to make zero-cost predictions
    def zero_cost_model(self,X,y,add_to_model=False):
        if self.base_model._estimator_type=='classifier':
            model = dummy.DummyClassifier("prior") 
        elif self.base_model._estimator_type=='regressor':
            model = dummy.DummyRegressor("mean")
        else: raise TypeError("sklearn Classifier or Regressor required!")
        cost = 0
        features = []
        model.fit(self.selectfeats(X,features),y)
        if add_to_model:
            self.model_costs.insert(0,cost)
            self.model_features.insert(0,features)
            self.models.insert(0,model)
        return (model, cost, features)

    # Build full model with all features
    def full_cost_model(self,X,y,add_to_model=False):
        model = self.newmodel()
        self.fitmodel(model,X,y)
        cost = np.sum(self.feature_costs)
        features = np.arange(X.shape[1])
        if add_to_model:
            self.model_costs.append(cost)
            self.model_features.append(features)
            self.models.append(model)
        return (model, cost, features)

    # Train intermediate cost models -- most work happens here (base doesn't do anything)
    def intermediate_cost_models(self,X,y,add_to_model=False):
        return [None]*3

    # Postcondition - models and model_costs are sorted by cost
    def fit(self,X,y,feature_costs=None,cost_criterion=None,**kwargs):
        self.X = X
        self.y = y
        if feature_costs is None:
            feature_costs = np.ones(X.shape[1])
        self.feature_costs = np.array(feature_costs)
        # import pdb; pdb.set_trace()
        self.fit_kwargs = kwargs
        if cost_criterion is None:
            cost_criterion = lambda x: True
        self.cost_criterion = cost_criterion

        # Add base model with 0 cost
        # print('zerocost')
        self.zero_cost_model(X,y,add_to_model=True)

        # Train full model for explanations and high-cost predictions
        # print('fullcost')
        full_model, full_cost, full_features = self.full_cost_model(X,y,add_to_model=False)
        self.full_model = full_model

        # Train intermediate cost models -- most work happens here (base doesn't do anything)
        # print('intermediatecost')
        self.intermediate_cost_models(X,y,add_to_model=True)

        # Last -- put full model at end
        # print('wrapup')
        self.models.append(full_model)
        self.model_costs.append(full_cost)
        self.model_features.append(full_features)

        # Ensure models are sorted
        inds = np.argsort(self.model_costs)
        self.model_costs = np.array([self.model_costs[i] for i in inds])
        self.models = [self.models[i] for i in inds]
        self.model_features = [self.model_features[i] for i in inds]
    
    # Get index of best possible model below max_cost
    def get_stop_index(self,max_cost):
        exceeds_max = np.array(self.model_costs)<=max_cost
        stop_index = np.nonzero(exceeds_max)[0][-1]
        return stop_index
        
    # Predict using best possible model for a given cost
    def predict(self,X,max_cost=np.inf):
        stop_index = self.get_stop_index(max_cost)
        model = self.models[stop_index]
        features = self.model_features[stop_index]
        return model.predict(self.selectfeats(X,features))
    
    # Predict_proba using best possible model for a given cost
    def predict_proba(self,X,max_cost=np.inf):
        if self.base_model._estimator_type!='classifier':
            raise TypeError("'predict_proba' only works with sklearn classifiers as base model!")
        stop_index = self.get_stop_index(max_cost)

        model = self.models[stop_index]
        features = self.model_features[stop_index]
        return model.predict_proba(self.selectfeats(X,features))

    # Score models with predict
    def score_models(self,Xtest,ytest,scorefunc):
        self.model_scores = np.array([scorefunc(ytest,self.predict(Xtest,max_cost=c)) 
               for c in self.model_costs])

    # Score models with predict_proba
    def score_models_proba(self,Xtest,ytest,scorefunc):
        self.model_scores = np.array([scorefunc(ytest,self.predict_proba(Xtest,max_cost=c)[:,1]) 
               for c in self.model_costs])

    # Feature attributions
    def explain_models(self,indiv=False):
        if indiv: all_attribs = np.zeros((len(self.models)-1,self.X.shape[1],self.X.shape[0]))
        else: all_attribs = np.zeros((len(self.models)-1,self.X.shape[1]))
        for i,m in iterator(enumerate(self.models[1:])):
            data = self.selectfeats(
                        self.X,
                        self.model_features[i+1])
            exp = self.base_explainer(m, data)
            attribs = exp.shap_values(data)
            if indiv: 
                if attribs.shape[1]==all_attribs.shape[1]:
                    all_attribs[i]=attribs.T
                else:
                    all_attribs[i][self.model_features[i+1]]=attribs.T
            else: 
                if attribs.shape[1]==all_attribs.shape[1]:
                    all_attribs[i]=np.mean(np.abs(attribs),axis=0)
                else:
                    all_attribs[i][self.model_features[i+1]] = np.mean(np.abs(attribs),axis=0)
        if indiv:
            self.indiv_attribs = all_attribs
            self.global_attribs = np.mean(np.abs(all_attribs),axis=2)
        else:
            self.global_attribs = all_attribs
                
    # Serialize model
    def save_model(self,fname):
        with open(fname, 'wb') as w:
            dill.dump(self,w)
            
    # Area under cost performance curve
    def auaucc(self,normalize=False, max_score=None, max_cost=np.inf):
        model = self
        total = 0.0
        for i in range(1,len(model.models)):
            if model.model_costs[i]<=max_cost:
                total += model.model_scores[i]*(model.model_costs[i]-model.model_costs[i-1])
            else:
                total += model.model_scores[i]*(max_cost-model.model_costs[i-1])
                break
        if normalize:
            assert max_score, "Need max score if we're going to normalize!"
            total /= max_score * min(max_cost,model.model_costs[-1])
        return total
    
    # Slices of the cost-performance curve
    def perf_at_cost(model,cost):
        return np.max(model.model_scores[model.model_costs<=cost])

    def cost_at_perf(model,perf):
        return np.min(model.model_costs[model.model_scores>=perf])
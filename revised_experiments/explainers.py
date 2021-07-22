from shap import TreeExplainer
import sage
import numpy as np
import pandas as pd

class OneDimExplainer(TreeExplainer):
    def __init__(self,model,data=None,**kwargs):
        super().__init__(model,data,**kwargs)
    def shap_values(self,data,y=None,**kwargs):
        shaps = super().shap_values(data,y,**kwargs)
        if type(shaps)==list:
            shaps = shaps[-1]
        return shaps

class OneDimDependent(TreeExplainer):
    def __init__(self,model,data,**kwargs):
        super().__init__(model,**kwargs)
    def shap_values(self,data,y=None,**kwargs):
        shaps = super().shap_values(data,y,**kwargs)
        if type(shaps)==list:
            shaps = shaps[-1]
        return shaps
    
class Loss1DExplainer(OneDimExplainer):
    def __init__(self,model, data,model_output='log_loss', feature_dependence='interventional',**kwargs):
        super().__init__(model,data,model_output=model_output,feature_dependence=feature_dependence,**kwargs)
        
def get_treeloss_wrapper(ytv,nsamp=512):
    class L1E(Loss1DExplainer):
        def __init__(self,model,data,**kwargs):
            d2 = data[:nsamp] if type(data)==np.ndarray else data.iloc[:nsamp]
            super().__init__(model,d2,**kwargs)
        def shap_values(self,X,**kwargs):
            return super().shap_values(X,ytv,**kwargs)
    return L1E

def get_pipeline_explainer(explainer):
    class PipelineWrapper(explainer):
        def __init__(self,model,data,*args,**kwargs):
            self.preprocessor = preprocessor = model[:-1]
            predictor = model[-1]
            data = preprocessor.transform(data)
            super().__init__(predictor,data,*args,**kwargs)
        def shap_values(self,data,*args,**kwargs):
            data = self.preprocessor.transform(data)
            return super().shap_values(data,*args,**kwargs)
    return PipelineWrapper

def df_default(df):
    if type(df)==np.ndarray: return np.nanmean(df,axis=0)
    categorical = np.array([str(x)=='category' for x in df.dtypes])
    default = df.iloc[[0]]
    default.iloc[0,~categorical] = np.nanmean(df.values[:,~categorical],axis=0)
    default.iloc[0,categorical] = [df[c].dropna().mode().values[0] for c in df.iloc[:,categorical].columns]
#     default = pd.DataFrame
    return default.values.flatten()

class SageTranslator(object):
    def __init__(self,model,df,lclip=1e-6,uclip=1-1e-6):
        self.model = model
        self.default = self.get_default(df)
        self.lclip, self.uclip = lclip, uclip
    def get_default(self,df):
        categorical = np.array([str(x)=='category' for x in df.dtypes])
        default = df.iloc[[0,0]].copy()
        default.iloc[0,~categorical] = np.nanmean(df.values[:,~categorical],axis=0)
        default.iloc[0,categorical] = [df[c].dropna().mode().values[0] for c in df.iloc[:,categorical].columns]
        return default.iloc[[0]]
    def wrap_df(self,df):
        x = df.copy() if type(df)==type(self.default) else pd.DataFrame(df,columns=self.default.columns)
        for c in x.columns: x[c] = x[c].astype(self.default[c].dtype)
        return x
    def predfunc(self,x):
        x = self.wrap_df(x)
        raw_preds = self.model.predict_proba(x)[:,1]
        return np.clip(raw_preds,self.lclip,self.uclip)
        
def get_sage_wrapper(ext_labels,groups=None,subsize=512,batch_size=16,imputetype='default',esttype=sage.PermutationEstimator,**kwargs):
    if type(ext_labels)==pd.core.frame.DataFrame: ext_labels = ext_labels.values
    class NNSageExplainer(object):
        def __init__(self,model,X):
            if type(X)==pd.core.frame.DataFrame: X = X.values
            self.predfunc = lambda x: np.clip(model.predict_proba(x)[:,1],1e-6,1-1e-6)
            if imputetype=='default': self.imputer = sage.DefaultImputer(self.predfunc,np.nanmean(X,axis=0))
#             assert type(X)==type(ext_labels)==np.ndarray, "Numpy arrays required!"
            else:
                subinds = np.random.choice(X.shape[0],size=subsize)
                subX = X[subinds]
                self.imputer = sage.MarginalImputer(self.predfunc,subX) if groups is None else sage.GroupedMarginalImputer(self.predfunc,subX,groups,remaining_features=X.shape[1]-np.unique([idx for g in groups for idx in g]).shape[0])
            self.estimator = esttype(self.imputer,'cross entropy')
            self.X = X
            self.y = ext_labels
        def shap_values(self,X):
            if type(X)==pd.core.frame.DataFrame: X = X.values
            assert self.X.shape==X.shape
#             np.testing.assert_equal(self.X,X), "Currently NNSageExplainer only works with same X,y passed to init and shap_values!"
            shaps = self.estimator(X,self.y,batch_size=batch_size,**kwargs)
            return shaps.values.reshape(1,-1)
    return NNSageExplainer

    
def icu_sage_wrapper(ext_labels,groups=None,subsize=512,batch_size=16,imputetype='default'):
    if type(ext_labels)==pd.core.frame.DataFrame: ext_labels = ext_labels.values
    class NNSageExplainer(object):
        def __init__(self,model,X):
            self.translator = SageTranslator(model,X)
            self.predfunc = self.translator.predfunc#lambda x: np.clip(model.predict_proba(x)[:,1],1e-6,1-1e-6)
            if type(X)==pd.core.frame.DataFrame: X = X.values
            if imputetype=='default': self.imputer = sage.DefaultImputer(self.predfunc,self.translator.default.values.flatten())
#             assert type(X)==type(ext_labels)==np.ndarray, "Numpy arrays required!"
            else:
                subinds = np.random.choice(X.shape[0],size=subsize)
                subX = X[subinds] if type(X)==np.ndarray else X.iloc[subinds]
                self.imputer = sage.MarginalImputer(self.predfunc,subX) if groups is None else sage.GroupedMarginalImputer(self.predfunc,subX,groups,remaining_features=X.shape[1]-np.unique([idx for g in groups for idx in g]).shape[0])
            self.estimator = sage.PermutationEstimator(self.imputer,'cross entropy')
            self.X = X
            self.y = ext_labels
        def shap_values(self,X):
            if type(X)==pd.core.frame.DataFrame: X = X.values
            assert self.X.shape==X.shape
#             np.testing.assert_equal(self.X,X), "Currently NNSageExplainer only works with same X,y passed to init and shap_values!"
            shaps = self.estimator(X,self.y,batch_size=batch_size)
            return shaps.values.reshape(1,-1)
    return NNSageExplainer
        
def labelless_sage_wrapper(groups=None,wrap_categorical=False,refsize=512,expsize=None,batch_size=16,imputetype='default',esttype=sage.PermutationEstimator, **kwargs):
    #if type(ext_labels)==pd.core.frame.DataFrame: ext_labels = ext_labels.values
    class NNSageExplainer(object):
        def __init__(self,model,X):
            if wrap_categorical:
                self.translator = SageTranslator(model,X)
                self.predfunc = self.translator.predfunc#lambda x: np.clip(model.predict_proba(x)[:,1],1e-6,1-1e-6)
            else: self.predfunc = lambda x: np.clip(model.predict_proba(x)[:,1],1e-6,1-1e-6)
            if type(X)==pd.core.frame.DataFrame: X = X.values
            if imputetype=='default': self.imputer = sage.DefaultImputer(self.predfunc,np.nanmean(X,axis=0))
#             assert type(X)==type(ext_labels)==np.ndarray, "Numpy arrays required!"
            else:
                subinds = np.random.choice(X.shape[0],size=refsize)
                subX = X[subinds]
                self.imputer = sage.MarginalImputer(self.predfunc,subX) if groups is None else sage.GroupedMarginalImputer(self.predfunc,subX,groups,remaining_features=X.shape[1]-np.unique([idx for g in groups for idx in g]).shape[0])
            self.estimator = esttype(self.imputer,'cross entropy')
        def shap_values(self,X,y):
            if type(X)==pd.core.frame.DataFrame: X = X.values
            if expsize is not None:
                inds = np.random.choice(X.shape[0],size=expsize)
                X,y = X[inds], y[inds]
#             np.testing.assert_equal(self.X,X), "Currently NNSageExplainer only works with same X,y passed to init and shap_values!"
            shaps = self.estimator(X,y,batch_size=batch_size,**kwargs)
            return shaps.values.reshape(1,-1)
    return NNSageExplainer
        
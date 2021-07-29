import pandas as pd
import config
import numpy as np
from sklearn.model_selection import train_test_split
from tuning import bootstrap_set

DATADIR = config.ED_DPATH_PROCESSED
SURVEYPATH = config.ED_SURVEYPATH

# One-indexed
def cost_pair(i,j,X):
    assert i>0 and j>0, "i and j are 1-indexed!"
    data = pd.read_csv(config.ED_INDIV_PATH,index_col=0)
    full_inds = data.isna().mean(1)==0
    v1 = data[full_inds].iloc[i-1].values
    v2 = data[full_inds].iloc[j-1].values
    cd1 = {k:v for k,v in zip(data.columns,v1)}
    cd2 = {k:v for k,v in zip(data.columns,v2)}
    lowcost = config.ED_LOWCOST
    costs1 = np.array([cd1[c] if c in cd1 else lowcost for c in X.columns])
    costs2 = np.array([cd2[c] if c in cd2 else lowcost for c in X.columns])
    return costs1, costs2

def load_ed(name,costtype,drop_redundant, split_seed=None):
    # Used for all other methods
    Xtrain = pd.read_pickle(f'{DATADIR}{name}-train-raw').iloc[:,:-1]
    Xvalid = pd.read_pickle(f'{DATADIR}{name}-val-raw').iloc[:,:-1]
    Xtest = pd.read_pickle(f'{DATADIR}{name}-test-raw').iloc[:,:-1]

    ytrain = pd.read_pickle(f'{DATADIR}{name}-train-raw').iloc[:,-1]
    yvalid = pd.read_pickle(f'{DATADIR}{name}-val-raw').iloc[:,-1]
    ytest = pd.read_pickle(f'{DATADIR}{name}-test-raw').iloc[:,-1]
    
    X = pd.concat([Xtrain,Xvalid,Xtest])
    y = np.hstack([ytrain,yvalid,ytest])
    
    if split_seed is not None:
        Xtv, Xtest, ytv, ytest = train_test_split(X,y,train_size=0.8,random_state=split_seed)
        Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xtv,ytv,train_size=0.8,random_state=split_seed)
        Xtrain.index, Xvalid.index, Xtest.index = [list(range(df.shape[0])) for df in (Xtrain,Xvalid,Xtest)]
    
    with open(SURVEYPATH,'r') as f:
        header = f.readline()
        cost_dict = {line.split(',')[0]: float(line.split(',')[4]) for line in f}
    
    lowcost = config.ED_LOWCOST
    costs = np.array([cost_dict[c] if c in cost_dict else lowcost for c in Xtrain.columns])
    
    dropped_inds = np.array(['protectivedevice' in c or c in ('scenelowestbloodpressure','scenegcs','scenehighestgcs') for c in Xtrain.columns])
    if drop_redundant:
        Xtrain = Xtrain.iloc[:,~dropped_inds]
        Xvalid = Xvalid.iloc[:,~dropped_inds]
        Xtest = Xtest.iloc[:,~dropped_inds]
        costs = costs[~dropped_inds]
    
    return (Xtrain,ytrain), (Xvalid,yvalid), (Xtest,ytest), costs, np.arange(Xtrain.shape[1]), {}

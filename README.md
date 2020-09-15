# CoAI

Main package with tools to make any predictive model cost-aware.
To install, run `python setup.py install` from the main directory; `python setup.py install --user` for local install.

For more examples, see `experiments`, in particular `experiments/ED Trauma.ipynb` and `experiments/NHANES.ipynb`.

Basic Usage:
```
from coai import GreedyOptimizer
import numpy as np
import shap
import xgboost as xgb

Xtrain = np.random.random((100,10))
Xtest = np.random.random((100,10))
ytrain = np.random.choice([0,1],size=100)
ytest = np.random.choice([0,1],size=100)

costs = np.arange(10)+1
cost_threshold = 2.5

bst = xgb.XGBClassifier(tree_method='approx',silent=1)
explainer = shap.TreeExplainer
GO = GreedyOptimizer(bst,explainer)
GO.fit(Xtrain,ytrain,costs)
GO.predict(Xtest,max_cost=cost_threshold)
```

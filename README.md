# CoAI

Main package with tools to make any predictive model cost-aware.
To install, run `python setup.py install` from the main directory; `python setup.py install --user` for local install.

For more examples, see `experiments`, in particular `experiments/ED Trauma.ipynb` and `experiments/NHANES.ipynb`.

Basic Usage:
```python
from coai import CoAIOptimizer
import numpy as np
import shap
import xgboost as xgb

# Random data
Xtrain = np.random.random((100,10))
Xtest = np.random.random((100,10))
ytrain = np.random.choice([0,1],size=100)
ytest = np.random.choice([0,1],size=100)

# Features fall into two groups, each with a randmly assigned cost
costs = np.hstack(([np.random.random()]*5,[np.random.random()]*5))
groups = np.hstack(([1]*5,[2]*5))

# Train models at three cost thresholds
thresholds = np.array([0.5,1.25,1.75])

# Base model and explainer
bst = xgb.XGBClassifier(tree_method='approx',silent=1)
explainer = shap.TreeExplainer

# Train CoAI
CoAI = CoAIOptimizer(bst,explainer,scale_ints=10000)
CoAI.fit(Xtrain,ytrain,
    feature_costs=costs,
    feature_groups=groups,
    thresholds=thresholds)

# Generate predictions with budget 1.25
deployment_threshold=1.25
preds = CoAI.predict(Xtest,max_cost=deployment_threshold)
```

import numpy as np
from coai import base
import sys
from explainers import labelless_sage_wrapper

dname, rseed, m_ind = sys.argv[1:]
rseed, m_ind = int(rseed), int(m_ind)

gbm = base.load_model(f'rebuttal_runs/{dname}-gbmsage-{rseed}-None.coai')

i=m_ind
model = gbm.models[i]
data = gbm.selectfeats(gbm.X,gbm.model_features[i])
exp_class = labelless_sage_wrapper(imputetype='marginal',refsize=64,batch_size=32,wrap_categorical=(dname=='icu'),n_permutations=None)
exp = exp_class(model,data)
attribs = exp.shap_values(data,gbm.y)

np.save(f'rebuttal_runs/{dname}-gbmsage-{rseed}-shaps{m_ind}.npy',attribs)

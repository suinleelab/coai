from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import numpy as np

class FastLinearClassifier(LogisticRegression):
    def predict_proba(self,X):
        if 'pandas.' in str(type(X)): X = X.values
        rawpreds = expit(X@self.coef_.flatten()+self.intercept_)
        return np.vstack((1-rawpreds,rawpreds)).T
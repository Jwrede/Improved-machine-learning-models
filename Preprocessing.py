import numpy as np
from sklearn.datasets import make_classification

def feature_normalize(X):
    X_norm = X.copy()
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    
    X_norm[:,1:] = (X[:,1:] - mu[1:]) / sigma[1:]
    
    return X_norm, mu, sigma

def create_poly_features(X, options):
    options = sorted(options, key = lambda x:x[0])
    options = [tuple(map(lambda x:abs(x), option)) for option in options]
    options = list(set(options))
    X_new = X[:,:options[0][0]]
    temp = 0
    
    for option in options:
        if temp:
            X_new = np.c_[X_new, X[:, temp:option[0] - 1], X[:, option[0]]]
        else:
            X_new = np.c_[ X[:, temp:option[0]], X[:, option[0]]]
        for i in range(2, option[1] + 2):
            X_new = np.c_[X_new, X[:,option[0]]**(i)]
        
        temp = option[0]
            
    X_new = np.c_[X_new, X[:, options[len(options) - 1][0] + 1:]]
    return X_new
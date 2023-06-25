import numpy as np

class MY_PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.scale = None
    
    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        X = (X - self.mean) / self.scale
        
        cov = np.cov(X.T)
        
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:,idx]
        self.components = eigenvectors[:, :self.n_components]       
        
        return np.dot(X, self.components)

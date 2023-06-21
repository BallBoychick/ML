import numpy as np

class MY_PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.scale = None
    
    def fit_transform(self, X):
        # вычисляем средние значения по столбцам
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        # центрируем данные
        X = (X - self.mean) / self.scale
        
        # вычисляем ковариационную матрицу
        cov = np.cov(X.T)
        
        # вычисляем собственные векторы и собственные значения ковариационной матрицы
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # выбираем k главных компонент
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:,idx]
        self.components = eigenvectors[:, :self.n_components]       
        
        # проецируем данные на главные компоненты
        return np.dot(X, self.components)
    # def transform(self, X):
    #     # центрируем данные
    #     X = X - self.mean
        
    #     # проецируем данные на главные компоненты
    #     return np.dot(X, self.components)
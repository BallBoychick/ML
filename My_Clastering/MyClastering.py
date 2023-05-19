import numpy as np
class MyKMeans:
    def __init__(self, n_clusters, tol, max_iter):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.n_clusters), :]

        for i in range(self.max_iter):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            print("I", i)
            new_centroids = []
            for j in range(self.n_clusters):
                new_centroid = X[labels == j].mean(axis=0)
                new_centroids.append(new_centroid)
            new_centroids = np.array(new_centroids)

            if np.abs(self.centroids - new_centroids).sum() < self.tol:
                break

            self.centroids = new_centroids
            print(self.centroids)

        self.labels = labels

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

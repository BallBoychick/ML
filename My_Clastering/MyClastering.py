import numpy as np
class KMeans:
    def __init__(self, n_clusters=3, tol=0.001, max_iter=300):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.n_clusters), :]

        for i in range(self.max_iter):
            # Assign each sample to the nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            print("I", i)
            # Update centroids based on the new assignment of samples


            # new_centroids = []
            # for j in range(self.n_clusters):
            #     if np.sum(labels == j) > 0:
            #         new_centroid = X[labels == j].mean(axis=0)
            #     else:
            #         new_centroid = self.centroids[j]
            #     new_centroids.append(new_centroid)
            # new_centroids = np.array(new_centroids)
            new_centroids = []
            for j in range(self.n_clusters):
                new_centroid = X[labels == j].mean(axis=0)
                new_centroids.append(new_centroid)
            new_centroids = np.array(new_centroids)

            # new_centroids = np.array([X[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else self.centroids[j] for j in range(self.n_clusters)])

            # Check for convergence
            if np.abs(self.centroids - new_centroids).sum() < self.tol:
                break

            self.centroids = new_centroids
            print(self.centroids)

        self.labels = labels

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

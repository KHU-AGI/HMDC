import torch
import torch.nn.functional as F

class KMeans:
    def __init__(self, n_clusters, mode='euclidean', max_iter=100, verbose=False, init='kmeans++'):
        self.n_clusters = n_clusters
        self.mode = mode
        self.init = init
        self.verbose = verbose
        self.max_iter = max_iter

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def fit(self, X):
        self.X = X
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.centroids = torch.zeros(self.n_clusters, self.n_features, device=self.X.device)
        self.labels = torch.zeros(self.n_samples, dtype=torch.long, device=self.X.device)
        self._init_centroids()
        for i in range(self.max_iter):
            old_centroids = self.centroids.clone()
            self._update_labels()
            self._update_centroids()
            if self._has_converged(old_centroids, self.centroids):
                break

    def predict(self, X):
        dist_matrix = self._compute_dist_matrix(X)
        return torch.argmin(dist_matrix, dim=1)
    
    def _init_centroids(self):
        if self.init == 'random':
            self.centroids = self.X[torch.randperm(self.n_samples)[:self.n_clusters]]
        elif self.init == 'kmeans++':
            _centroid_buffer = self.centroids
            self.centroids = _centroid_buffer[:1]
            self.centroids[0] = self.X[torch.randint(self.n_samples, (1,))]
            for i in range(1, self.n_clusters):
                dist_matrix = self._compute_dist_matrix(self.X)
                dist = torch.min(dist_matrix, dim=1).values + 1e-12
                dist = dist / dist.sum()
                idx = torch.multinomial(dist, 1)
                _centroid_buffer[i] = self.X[idx]
                self.centroids = _centroid_buffer[:i+1]

    def _update_labels(self):
        dist_matrix = self._compute_dist_matrix(self.X)
        self.labels = torch.argmin(dist_matrix, dim=1)

    def _update_centroids(self):
        for i in range(self.n_clusters):
            in_i = self.labels == i
            if in_i.sum() == 0:
                continue
            self.centroids[i] = torch.mean(self.X[self.labels == i], dim=0)

    def _compute_dist_matrix(self, X):
        dist_matrix = self._compute_dist(X, self.centroids, self.mode)
        return dist_matrix
    
    def _compute_dist(self, X, y, mode):
        X = X.view(X.shape); y = y.view(y.shape);
        if mode == 'euclidean':
            # return self._euclidean_dist(X, y)
            return torch.cdist(X, y, p=2)
        elif mode == 'cosine':
            return F.cosine_similarity(X.unsqueeze(0), y.unsqueeze(1), dim= -1)
            # return self._cosine_dist(X, y)
        else:
            raise NotImplementedError
        
    def _euclidean_dist(self, X, y):
        m, n = X.size(0), y.size(0)
        xx = torch.pow(X, 2).sum(1, keepdim=True).repeat(1, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).repeat(1, m).t()
        dist = xx + yy
        dist = dist.addmm(X, y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()
        return dist
    
    def _cosine_dist(self, X, y):
        X_norm = F.normalize(X, dim=1)
        y_norm = F.normalize(y, dim=1)
        return (1 - torch.mm(X_norm, y_norm.t())).clamp(min=1e-12)
    
    def _compute_inertia(self):
        dist_matrix = self._compute_dist_matrix(self.X)
        return torch.sum(torch.min(dist_matrix, dim=1)[0])
    
    def _has_converged(self, old_centroids, centroids):
        return torch.equal(old_centroids, centroids)
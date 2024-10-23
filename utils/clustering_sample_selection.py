import torch
from utils.kmeans import KMeans

class Clustering_Sample_Selection:
    def __init__(self, image, net_fn, mode='euclidean'):
        self.net_fn = net_fn
        self.image = image
        self.mode = mode

    def query(self, batchsize=128):
        with torch.no_grad():
            for i in range(0, len(self.image), batchsize):
                embeddings = self.net_fn(self.image[i:i+batchsize])
                embeddings = embeddings.flatten(1)
                if i == 0: all_embeddings = embeddings
                else: all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
        kmeans = KMeans(n_clusters=batchsize, max_iter=100, batchsize=batchsize, mode=self.mode, init='kmeans++', seed=None)
        labels = kmeans.fit_predict(all_embeddings).clone()
        dist_matrix = kmeans.compute_distance_matrix(all_embeddings, kmeans.get_centroids().clone())
        dist = dist_matrix[torch.arange(len(dist_matrix)), labels]
        dist = dist / dist.max()
        q_idxs = torch.argsort(dist_matrix,dim=0).t()
        selected_idxs = []
        for idxes in q_idxs:
            for idx in idxes:
                if idx not in selected_idxs:
                    selected_idxs.append(idx)
                    break;
            if len(selected_idxs) == batchsize:
                break
        selected_idxs = torch.tensor(selected_idxs).to(q_idxs.device)
        return selected_idxs, dist.mean()
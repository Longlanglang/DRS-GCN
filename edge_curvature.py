from GraphRicciCurvature.OllivierRicci import OllivierRicci
import torch, time
from torch_geometric.utils import to_networkx


device = "cuda" if torch.cuda.is_available() else "cpu"
class GetEdgeCurvature:
    def __init__(self, data):
        self.data = data

    def compute_ricci_curvature(self):
        """
        Compute the Ricci curvature of a graph
        :return: Ricci curvature
        """
        G = to_networkx(self.data)
        ricci_curvature = OllivierRicci(G)
        G = ricci_curvature.compute_ricci_curvature()
        ricci_curvature_dict = {}
        for source, target, data in G.edges(data=True):
            if 'ricciCurvature' in data:
                ricci_curvature_dict[(source, target)] = data['ricciCurvature']

        ricci_weight_list = torch.zeros(self.data.edge_index.shape[1], device=device)
        for i in range(self.data.edge_index.shape[1]):
            source, target = self.data.edge_index[0, i].item(), self.data.edge_index[1, i].item()

            if (source, target) in ricci_curvature_dict:
                ricci_weight_list[i] = torch.tensor(ricci_curvature_dict[(source, target)])


        return ricci_weight_list

class GetEdgeWeight:
    def __init__(self, feature, ricci_curvature, edge_index):
        self.feature = feature
        self.ricci_curvature = ricci_curvature
        self.edge_index = edge_index

    def get_similarity_matrix(self):
        representation_normalized = self.feature / (self.feature.norm(2, 1).reshape(-1, 1) + 1e-10)
        representation_normalized = representation_normalized.to(device)
        similarity_matrix = torch.mm(representation_normalized, representation_normalized.T)
        similarity_matrix = similarity_matrix.to(device)
        source_nodes = self.edge_index[0]
        target_nodes = self.edge_index[1]

        edge_similarity_weight = similarity_matrix[source_nodes, target_nodes]

        return edge_similarity_weight

    def compute_edge_weight(self):
        """
        Compute the weight of edge for reorganization
        :return:
        """
        edge_similarity_weight = self.get_similarity_matrix().to(device)
        result = (edge_similarity_weight) * torch.exp(-2 * (self.ricci_curvature) ** 2)
        return result

class WeightDrop:
    def __init__(self, edge_index, edge_weight):
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def weight_drop(self, percent):
        assert 0 <= percent <= 1, "Percent should be between 0 and 1."

        num_edges = self.edge_index.size(1)
        num_preserve = int(num_edges * percent)

        _, indices = self.edge_weight.sort(descending=True)
        indices = indices[: num_preserve]
        edge_index_sampled = self.edge_index[:, indices].to(device)
        edge_weight_sampled = self.edge_weight[indices].to(device)

        return edge_index_sampled, edge_weight_sampled

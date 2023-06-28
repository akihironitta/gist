import torch

from torch_geometric.datasets import GDELTLite
from torch_geometric.nn.models.graph_mixer import LinkEncoder, NodeEncoder


class GraphMixer(torch.nn.Module):
    def __init__(
        self,
        num_edge_feats: int,
    ) -> None:
        super().__init__()
        self.link_encoder = LinkEncoder(
            k=30,
            in_channels=num_edge_feats,
            hidden_channels=12,
            out_channels=34,
            time_channels=56,
            is_sorted=False,
            dropout=0.2,
        )
        self.node_encoder = NodeEncoder(time_window=78)
        self.mlp = torch.nn.Linear(2 * 100, 100)

    def forward(self, x, edge_index, edge_attr, edge_time, seed_time):
        link_feat = self.link_encoder(
            edge_index,
            edge_attr,
            edge_time,
            seed_time,
        )  # [num_nodes, out_channels]
        node_feat = self.node_encoder(
            x,
            edge_index,
            edge_time,
            seed_time,
        )  # [num_nodes, ]
        print(link_feat.size(), node_feat.size())
        feats = torch.cat([link_feat, node_feat], dim=-1)
        print(feats.size())
        feat_i = feats.unsqueeze(1)
        feat_j = feats.unsqueeze(0)
        feats = torch.cat([feat_i, feat_j], dim=-1)
        return feats


def main():
    num_nodes, num_edges = 10, 20
    num_feats, num_edge_feats = 3, 5

    x = torch.rand(num_nodes, num_feats)
    edge_index = torch.randint(high=num_nodes, size=(2, num_edges))
    edge_attr = torch.rand(num_edges, num_edge_feats)
    edge_time = torch.randint(100, (num_edges,))
    seed_time = torch.zeros_like(edge_time)

    assert x.size() == (num_nodes, num_feats)
    assert edge_index.size() == (2, num_edges)
    assert edge_attr.size() == (num_edges, num_edge_feats)
    assert edge_time.size() == seed_time.size() == (num_edges,)

    model = GraphMixer(
        num_edge_feats,
    )
    out = model(x, edge_index, edge_attr, edge_time, seed_time)
    print(out)


if __name__ == "__main__":
    main()

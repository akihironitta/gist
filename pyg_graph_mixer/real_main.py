import torch


import torch_geometric
from torch_geometric.datasets import GDELTLite
from torch_geometric.nn.models.graph_mixer import LinkEncoder, NodeEncoder


class GraphMixer(torch.nn.Module):
    def __init__(
        self,
        num_node_feats: int,
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
        self.link_classifier = torch.nn.Linear((34 + num_node_feats) * 2, 1)

    def forward(self, x, edge_index, edge_attr, edge_time, seed_time):
        # [num_nodes, out_channels]
        link_feat = self.link_encoder(
            edge_index,
            edge_attr,
            edge_time,
            seed_time,
        )

        # [num_nodes, num_node_feats]
        node_feat = self.node_encoder(
            x,
            edge_index,
            edge_time,
            seed_time,
        )

        # [num_nodes, out_channels + num_node_feats]
        feats = torch.cat([link_feat, node_feat], dim=-1)

        # [num_pos_pairs + num_neg_pairs, 1]
        out = self.link_classifier(
            feats,
        )

        return out


def main():
    torch.set_default_device("cpu")  # or cuda

    # TODO: Modify GDELTLite to have the default splits
    data = GDELTLite("./")[0]

    # loader = LinkNeighborLoader(
    #     data,
    #     num_neighbors=[3, 3],
    #     neg_sampling_ratio=2.0,
    #     edge_label=data.edge_attr,
    #     time_attr="time",
    #     edge_label_time=data.time,
    #     batch_size=128,
    #     shuffle=False,
    # )
    train_loader = torch_geometric.loader.TemporalDataLoader()
    val_loader = torch_geometric.loader.TemporalDataLoader()
    train_loader = torch_geometric.loader.TemporalDataLoader()

    model = GraphMixer()

    for sampled_data in train_loader:



def test_graph_mixer():
    num_nodes, num_edges = 10, 20
    num_node_feats, num_edge_feats = 3, 5

    x = torch.rand(num_nodes, num_node_feats)
    edge_index = torch.randint(high=num_nodes, size=(2, num_edges))
    edge_attr = torch.rand(num_edges, num_edge_feats)
    edge_time = torch.rand(num_edges)
    seed_time = torch.ones(num_nodes)

    assert x.size() == (num_nodes, num_node_feats)
    assert edge_index.size() == (2, num_edges)
    assert edge_attr.size() == (num_edges, num_edge_feats)
    assert edge_time.size() == seed_time.size() == (num_edges,)

    model = GraphMixer(
        num_node_feats=num_node_feats,
        num_edge_feats=num_edge_feats,
    )
    out = model(x, edge_index, edge_attr, edge_time, seed_time)
    print(out)


if __name__ == "__main__":
    main()
    # test_graph_mixer()

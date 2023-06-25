import os
import torch
import torch_geometric
import pandas as pd
import numpy as np


class GDELTLite(
    torch_geometric.datasets.icews.EventDataset
):  # TODO: check InMemoryDataset
    r"""The Global Database of Events, Language, and Tone (GDELT) dataset used
    in `"TGL: A General Framework for Temporal GNN Training on Billion-Scale
    Graphs" <http://arxiv.org/abs/2203.14883>`_, consisting of
    events collected from 2016 to 2020.

    Each node (actor) has:
    - 413-dimensional multi-hot feature vector that represents CAMEO codes attached to the corresponding actor to server

    Each edge (event) has:
    - timestamp
    - 186-dimensional multi-hot vector representing CAME codes attached to the corresponding event to server

    Tasks:
    - link prediction task is to predict whether there will be an event between two actors at a timestamp (for GraphMixer)
    - TODO: node classification task is to predict countries where the actors were located when the events happened (from TGL paper)

    splits:
    - training: before 2019
    - validation: 2019
    - testing: 2020

    GDELT 8,831   1,912,909 0.1 413 186 Yes Yes Yes

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10
        :header-rows: 1

        * - Node/Edge Type
          - #nodes/#edges
          - #features
          - #classes
        * - Actor
          - 8,831
          - 413
          -
        * - Event
          - 1,912,909
          - 186
          -
    """

    # TODO: Upload subsampled dataset somewhere to avoid downloading the whole dataset (41GB)
    url = "https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "GDELTLite", "raw")

    @property
    def raw_file_names(self) -> list[str]:
        return [
            "node_features.pt",
            "edges.csv",
            "edge_features.pt",
        ]

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "GDELTLite", "processed")

    @property
    def processed_file_names(self) -> list[str]:
        return "data.pt"

    def download(self):
        for filename in self.raw_file_names:
            print(filename)
            torch_geometric.data.download_url(
                f"{self.url}/{filename}", self.raw_dir
            )

    def process(self):
        # code adapted from:
        # https://github.com/CongWeilin/GraphMixer/blob/bbdd9923706a02d0619dfd00ef7880911f003a65/DATA/GDELT_lite/gen_dataset.py
        # FIXME: replace processed with raw
        # df = pd.read_csv(f"{self.raw_dir}/edges.csv")
        df = pd.read_csv(f"{self.processed_dir}/edges.csv")

        # GDELTLite is 1/100 of the original GDELT
        select = np.arange(0, len(df), 100)

        # edges
        new_df = pd.DataFrame(
            data={
                "Unnamed: 0": np.arange(len(select)),
                "src": df.src.values[select],
                "dst": df.dst.values[select],
                "time": df.time.values[select],
                "int_roll": df.int_roll.values[select],
                "ext_roll": df.ext_roll.values[select],
            }
        )

        # create edge features
        # FIXME: replace processed with raw
        # edge_attr = torch.load(f"{self.raw_dir}/edge_features.pt")
        edge_attr = torch.load(f"{self.processed_dir}/edge_features.pt")
        edge_attr = edge_attr[select]
        print(edge_attr.shape)

        # create node features
        # FIXME: replace processed with raw
        # x = torch.load(f"{self.raw_dir}/node_features.pt")
        x = torch.load(f"{self.processed_dir}/node_features.pt")

        edge_index = torch.LongTensor(new_df[["src", "dst"]].T.values)
        edge_timestamp = torch.LongTensor(new_df.time.values)
        data = torch_geometric.data.Data(
            x=x,  # [num_nodes, 413]
            edge_index=edge_index,  # [2, num_edges]
            edge_attr=edge_attr,  # [num_edges, 186]
            edge_label=edge_attr,  # [num_edges, 186]
            edge_label_index=edge_index,  # [2, num_edges]
            time=edge_timestamp,  # [num_edges,]
        )
        self.save([data], self.processed_paths[0])


dataset = GDELTLite(root="./")
data = dataset[0]
print(data)
data.validate()
batch_size = 2
loader = torch_geometric.loader.LinkNeighborLoader(
    data=data,
    num_neighbors=[2, 2],
    batch_size=batch_size,
    edge_label_index=None,  # sample from all edges
    edge_label=data.edge_label,  # same length as `edge_label_index`
    edge_label_time=data.time,  # same length as `edge_label_index`
    temporal_strategy="last",
    time_attr="time",
    # directed=False,  # unsupported?
)

print("=============================================================")
data = next(iter(loader))
print(data)

print("input edge index", data.edge_index.size())  # []

# [2, num_sampled_edges]
print("output edge index", data.edge_label_index.size())
# [num_sampled_edges, 186]
print("output edge label", data.edge_label.size())
# [num_sampled_edges,]
print("edge timestamp", data.time.size())

num_sampled_edges = len(data.edge_index)
assert (
    num_sampled_edges >= batch_size
), "sampled edges are (almost) always larger than batch_size"


def model(x, edge_index, time) -> torch.Tensor:
    # === graph mixer ===
    # 1. encode link
    # 1.1. temporal encoding
    # 1.2. 1-layer MLP-mixer
    # 2. encode node
    # 3. classify link
    print(
        f"... model(x={x.size()}, edge_index={edge_index.size()}, time={time.size()})"
    )
    return torch.zeros((num_sampled_edges, 186))


def loss_fn(edge_label_out, edge_label) -> torch.Tensor:
    print(
        f"... loss_fn(edge_label_out={edge_label_out.size()}, edge_label={edge_label.size()})"
    )
    return torch.zeros((1,))


edge_label_out = model(data.x, data.edge_index, data.time)
loss = loss_fn(edge_label_out[:batch_size], data.edge_label[:batch_size])

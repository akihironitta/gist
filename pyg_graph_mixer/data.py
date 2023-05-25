import os
import torch
import torch_geometric
import pandas as pd
import numpy as np


class GDELTLite(torch_geometric.datasets.icews.EventDataset):
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
    - link prediction task is to predict whether there will be an event between two actors at a timestamp
    - node classification task is to predict whether

    splits:
    - training: before 2019
    - validation: 2019
    - testing: 2020
    """

    # TODO: Upload subsampled dataset to avoid downloading the whole dataset (41GB)
    url = "https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        x = torch.load(f"{self.processed_dir}/node_features.pt")
        print("x", x.size())
        edge_attr = torch.load(f"{self.processed_dir}/edge_features.pt")
        print("edge_attr", edge_attr.size())

        df = pd.read_csv(f"{self.processed_dir}/edges.csv")

        print("df", df)

    @property
    def num_nodes(self) -> int:
        return 23033  # FIXME

    @property
    def num_rels(self) -> int:
        return 256  # FIXME

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
        return [
            "node_features.pt",
            "edges.csv",
            "edge_features.pt",
        ]

    def download(self):
        for filename in self.raw_file_names:
            print(filename)
            torch_geometric.data.download_url(
                f"{self.url}/{filename}", self.raw_dir
            )

    def process(self):
        df = pd.read_csv(f"{self.raw_dir}/edges.csv")

        # GDELTLite is 1/100 of the original GDELT
        select = np.arange(0, len(df), 100)
        new_df = {
            "Unnamed: 0": np.arange(len(select)),
            "src": df.src.values[select],
            "dst": df.dst.values[select],
            "time": df.time.values[select],
            "int_roll": df.int_roll.values[select],
            "ext_roll": df.ext_roll.values[select],
        }

        # create edges.csv
        new_df = pd.DataFrame(data=new_df)
        new_df.to_csv(f"{self.processed_dir}/edges.csv", index=False)

        # create edge features
        edge_feats = torch.load(f"{self.raw_dir}/edge_features.pt")
        torch.save(
            edge_feats[select], f"{self.processed_dir}/edge_features.pt"
        )

        # create node features
        node_feats = torch.load(f"{self.raw_dir}/node_features.pt")
        torch.save(node_feats, f"{self.processed_dir}/node_features.pt")


dataset = GDELTLite(root="./")
# loader = torch_geometric.loader.NeighborLoader(
#     dataset[0], num_neighbors=[2, 2]
# )
# data = next(iter(loader))
# assert isinstance(data, torch_geometric.data.Batch)

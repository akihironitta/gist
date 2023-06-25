# https://github.com/amazon-science/tgl/blob/716e9955d6d9bd2f18862319e97c478f8f4ec510/down.sh
import os

from mlp_mixer_pytorch import MLPMixer
import numpy as np
import pandas as pd
import torch
import torch_geometric


class GraphMixer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        """
        Args:
            in_channels: asdf
            hidden_channels: asdf
            out_channels: adfs
        """
        d = 100
        super().__init__()

        # 1.1. time-encoding function
        self.temporal_encoder = torch_geometric.nn.TemporalEncoding(d)

        # 1.2.  mixer for information summarizing
        #       1-layer MLP-mixer to summarize temporal link info.
        self.mlp_mixer = torch.nn.Sequential([])
        # 2.
        self.node_encoder = ...  # FIXME

        # 3. link classifier

    def forward(self, x, edge_index, edge_attr) -> torch.Tensor:
        timestamps = ...  # input
        K = 3  # keep the top K most recent temporal link infor- mation, where K is a dataset dependent hyper-parameter

        # Encode timestamps by our time-encoding function# output: [num_edges, d]
        # e.g. t_1 -> cos((t_0 - t_1)w)
        # mapping: timestamps -> temporal_features
        temporal_features = self.link_encoder(timestamps)

        # concatenate it with its corresponding link features
        # we stack all the outputs into a big matrix and zero-pad to the fixed length K denoted as T_2(t0).
        # e.g. cos((t_0 - t_1)w), x_{1,2}(t_1) -> cos((t_0 - t_1)w), x_{1,2}(t_1)
        # output: [num_edges, d+edge_attr]
        # mapping: [temporal_features || edge_attr] ->
        T_2 = torch.cat((temporal_features, edge_attr))  # FIXME: zero-padding

        # MLP-mixer
        t_2 = self.mlp_mixer(T_2)


# model = SAGE(dataset.num_features, 256, dataset.num_classes).to(device)
# model = GraphMixer(dataset.num_features, 256, dataset.num_classes).to(device)
dataset = torch_geometric.datasets.GDELT(root="./")
print(len(dataset))

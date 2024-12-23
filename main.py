
import torch
import torch_geometric
import numpy as np
from torch_geometric.nn import GCNConv, MessagePassing
from torch.nn import Sequential as Seq, Linear

device = 'cuda:0'

class Directional_EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.mlp = Seq(Linear(2 * in_channels + 1, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        #self.directional_ft = direction_feature(edge_index)
        edge_ft = torch.unsqueeze(edge_index[0,:] - edge_index[1,:],1)
        #Shape: [2,E] edge_indices[0,:] has all other nodes connecting to edge_indices[1,:] (less often changing)
        for i in range(edge_ft.shape[0]):
            if edge_ft[i,0] < 0:
                edge_ft[i,0] = -1
            elif edge_ft[i,0] > 0:
                edge_ft[i,0] = 1

        self.directional_ft = edge_ft
        # propagate_type: (x: Tensor)
        return self.propagate(edge_index=edge_index, x=x, size=None)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        #Expand edge_feature for the entire batch of data
        edge_feature =  self.directional_ft.unsqueeze(0).repeat(x_i.shape[0],1,1)
        tmp = torch.cat([x_i, x_j - x_i, edge_feature], dim=-1)  # tmp has shape [E, 2 * in_channels + 1]
        #tmp = torch.cat([x_i, x_j - x_i], dim=-1)
        return self.mlp(tmp)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, depth=3, device="", nn_inputs = 1):
        super().__init__()
        torch.manual_seed(1234567)
        self.depth = depth
        self.convIn = Directional_EdgeConv(nn_inputs, hidden_channels).to(device)
        for i in range(1, depth * 2 + 1):
            self.add_module(f'GCNConv{i}',
                            Directional_EdgeConv(hidden_channels, hidden_channels).to(device))
        self.convOut = Directional_EdgeConv(hidden_channels,1).to(device)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x_2 =  self.convIn(x, edge_index)
        for i in range(1, self.depth * 2 + 1, 2):
            x_1 = torch.nn.LeakyReLU(inplace=True)(getattr(self, f'GCNConv{i}')(x_2, edge_index))
            x_2 = torch.nn.LeakyReLU(inplace=True)(getattr(self, f'GCNConv{i + 1}')(x_1, edge_index) + x_2)
        x_out = self.convOut(x_2, edge_index) + x
        return x_out

SN = 48
def I(i):
    return (i % SN)  # periodic

neighbour_range = 1
edge_index = []
for i in range(SN):
    edge_index.append([I(i), I(i)])
    for j in range(1, neighbour_range + 1):
        edge_index.append([I(i), I(i - j)])
        edge_index.append([I(i), I(i + j)])

edges_torch = torch.from_numpy( np.array(edge_index)).to(device).long()
depth_ = 3
net = GCN(hidden_channels=32,depth=depth_, device=device)
net = torch_geometric.compile(net)
op = net(torch.rand(128,48,1).to(device), edges_torch.transpose(0,1))

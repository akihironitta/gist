import torch
import torch_geometric

torch_geometric.backend.use_segment_matmul = False

device = 'cuda'
conv = torch_geometric.nn.RGCNConv(4, 32, 4, None, None, aggr='sum').to(device)
c_conv = torch.compile(conv)

x = torch.randn(4, 4, device=device)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 0, 1, 1, 2, 2, 3],
    [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
], device=device)
edge_type = torch.tensor(
    [0, 1, 1, 0, 0, 1, 2, 3, 3, 2, 2, 3],
    device=device,
)

out1 = conv(x, edge_index, edge_type)
out2 = c_conv(x, edge_index, edge_type)
assert torch.allclose(out1, out2, atol=1e-8)

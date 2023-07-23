from rich.console import Console
from rich.table import Table
import torch
from torch_geometric.datasets import GDELTLite, Planetoid
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T


HIDE_TENSOR = True


def _describe_data(
    data,
    attribute_dict,
    title="PyG Data Description",
    color="green",
):
    table = Table(title=title)
    table.add_column("name", justify="center", style=color, no_wrap=True)
    table.add_column("size", justify="left", style=color)
    table.add_column("known size", justify="left", style=color)
    table.add_column("type", justify="left", style=color)
    table.add_column("content", justify="left", style=color)

    for attr_name, attr_known_shape in attribute_dict.items():
        attr = getattr(data, attr_name, None)
        if attr is None:
            continue

        if isinstance(attr, torch.Tensor):
            size_str = str(list(attr.shape))
        elif isinstance(attr, list):
            size_str = str([len(attr)])
        elif isinstance(attr, int):
            size_str = "1"
        else:
            size_str = ""

        attr_str = str(attr)
        if HIDE_TENSOR and isinstance(attr, torch.Tensor):
            attr_str = "-"

        table.add_row(
            attr_name,
            size_str,
            "\\" + attr_known_shape
            if attr_known_shape[0] == "["
            else attr_known_shape,
            type(attr).__name__,
            attr_str,
        )
    console = Console()
    console.print(table)


def describe_data(data):
    print(data, data.is_undirected())

    ### Node-related attributes
    attribute_known_shapes = {
        "num_nodes": "1",
        "x": "[num_nodes, num_node_features]",
        "y": "[num_nodes]",
        "node_time": "[num_nodes] TOCHECK",
        "num_sampled_nodes": "[batch_size, num_neighbors]",
        "n_id": "[num_nodes]",
    }
    _describe_data(
        data, attribute_known_shapes, "node-related attributes", color="blue"
    )

    ### Edge-related attributes
    attribute_known_shapes = {
        "num_edges": "1",
        "edge_attr": "[num_edges, num_edge_features]",
        "edge_index": "[2, num_edges]",
        "edge_label": "[batch_size]",
        "edge_time": "[num_edges] TOCHECK",
        "edge_label_time": "[num_edges] TOCHECK",
        "edge_label_index": "[2, batch_size]",
        "num_sampled_edges": "[1]",
        "e_id": "[num_edges]",
    }
    _describe_data(
        data, attribute_known_shapes, "edge-related attributes", color="red"
    )
    attribute_known_shapes = {
        "input_id": "[batch_size]",
        "time": "TODO",
        "batch": "TODO",
    }
    _describe_data(
        data, attribute_known_shapes, "misc attributes", color="yellow"
    )


def main():
    # TODO: Split train/val/test
    data = GDELTLite("data")[0]
    # Data(x=[16682, 413], edge_index=[2, 1912909], edge_attr=[1912909, 182], edge_time=[1912909])
    K = 2
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[K],
        neg_sampling_ratio=0.0,
        # edge_label=torch.ones(data.edge_index.size(1)),
        time_attr="edge_time",
        edge_label_time=data.edge_time,
        batch_size=13,
        shuffle=True,
    )
    sample = next(iter(loader))
    describe_data(sample)

    dataset = Planetoid(
        "data",
        "Cora",
        transform=T.NormalizeFeatures(),
    )
    data = dataset[0]
    # Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
    train_loader = LinkNeighborLoader(
        data,
        batch_size=13,
        num_neighbors=[K],
        edge_label=torch.ones(data.edge_index.size(1)),
        # time_attr="edge_index",
        # edge_label_time=torch.randint_like(data.edge_index, 0, 1),
        shuffle=False,
    )
    sample = next(iter(train_loader))
    describe_data(sample)


if __name__ == "__main__":
    main()

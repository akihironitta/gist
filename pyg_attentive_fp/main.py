# https://github.com/pyg-team/pytorch_geometric/blob/2.3.1/examples/attentive_fp.py
# MIT License Copyright (c) 2023 PyG Team <team@pyg.org>
import argparse
import time
import os.path as osp
from math import sqrt

import torch
import torch.nn.functional as F
from rdkit import Chem

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP


class GenFeatures:
    def __init__(self):
        self.symbols = [
            "B",
            "C",
            "N",
            "O",
            "F",
            "Si",
            "P",
            "S",
            "Cl",
            "As",
            "Se",
            "Br",
            "Te",
            "I",
            "At",
            "other",
        ]

        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            "other",
        ]

        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]

    def __call__(self, data):
        # Generate AttentiveFP features according to Table 1.
        mol = Chem.MolFromSmiles(data.smiles)

        xs = []
        for atom in mol.GetAtoms():
            symbol = [0.0] * len(self.symbols)
            symbol[self.symbols.index(atom.GetSymbol())] = 1.0
            degree = [0.0] * 6
            degree[atom.GetDegree()] = 1.0
            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.0] * len(self.hybridizations)
            hybridization[
                self.hybridizations.index(atom.GetHybridization())
            ] = 1.0
            aromaticity = 1.0 if atom.GetIsAromatic() else 0.0
            hydrogens = [0.0] * 5
            hydrogens[atom.GetTotalNumHs()] = 1.0
            chirality = 1.0 if atom.HasProp("_ChiralityPossible") else 0.0
            chirality_type = [0.0] * 2
            if atom.HasProp("_CIPCode"):
                chirality_type[
                    ["R", "S"].index(atom.GetProp("_CIPCode"))
                ] = 1.0

            x = torch.tensor(
                symbol
                + degree
                + [formal_charge]
                + [radical_electrons]
                + hybridization
                + [aromaticity]
                + hydrogens
                + [chirality]
                + chirality_type
            )
            xs.append(x)

        data.x = torch.stack(xs, dim=0)

        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

            bond_type = bond.GetBondType()
            single = 1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0
            double = 1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0
            triple = 1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0
            aromatic = (
                1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0
            )
            conjugation = 1.0 if bond.GetIsConjugated() else 0.0
            ring = 1.0 if bond.IsInRing() else 0.0
            stereo = [0.0] * 4
            stereo[self.stereos.index(bond.GetStereo())] = 1.0

            edge_attr = torch.tensor(
                [single, double, triple, aromatic, conjugation, ring] + stereo
            )

            edge_attrs += [edge_attr, edge_attr]

        if len(edge_attrs) == 0:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_indices).t().contiguous()
            data.edge_attr = torch.stack(edge_attrs, dim=0)

        return data


def train(model, optimizer, dataloader, device):
    total_loss = total_examples = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    return sqrt(total_loss / total_examples)


def main(args):
    path = osp.join(
        osp.dirname(osp.realpath(__file__)), "..", "data", "AFP_Mol"
    )
    dataset = MoleculeNet(
        path, name="ESOL", pre_transform=GenFeatures()
    ).shuffle()

    N = len(dataset) // 10
    train_dataset = dataset[2 * N :]
    train_loader = DataLoader(
        train_dataset,
        batch_size=200,
        shuffle=True,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentiveFP(
        in_channels=39,
        hidden_channels=200,
        out_channels=1,
        edge_dim=10,
        num_layers=2,
        num_timesteps=2,
        dropout=0.2,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=10**-2.5, weight_decay=10**-5
    )

    if args.profile:
        with torch.profiler.profile(
            with_stack=True,
            record_shapes=args.record_shapes,
        ) as p:
            for epoch in range(5):
                _ = train(model, optimizer, train_loader, device)
        p.export_chrome_trace("profile.trace.json")
    else:
        for epoch in range(1, 21):
            t0 = time.time()
            train_rmse = train(model, optimizer, train_loader, device)
            print(
                f"Epoch: {epoch:03d}, "
                f"Loss: {train_rmse:.4f}, "
                f"Time: {(time.time() - t0)*1_000:>6.1f} ms"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--record_shapes", action="store_true")
    main(parser.parse_args())

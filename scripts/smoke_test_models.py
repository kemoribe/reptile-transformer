"""Run one synthetic forward pass through every published model entry."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
GRAPHDTA_ROOT = ROOT / "graphdta"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GRAPHDTA_ROOT))

from run_baseline_mlp import BaselineMLP  # noqa: E402
import reptile_transformer_model as transformer_module  # noqa: E402
from models.gat import GATNet  # noqa: E402
from models.gat_gcn import GAT_GCN  # noqa: E402
from models.gcn import GCNNet  # noqa: E402
from models.ginconv import GINConvNet  # noqa: E402


def check_vector_model(name: str, model: torch.nn.Module) -> None:
    model.eval()
    model_device = next(model.parameters()).device
    batch_size = 2
    inputs = (
        torch.rand(batch_size, 2048, device=model_device),
        torch.zeros(batch_size, 167, device=model_device),
        torch.rand(batch_size, 10, device=model_device),
        torch.rand(batch_size, 480, device=model_device),
    )
    with torch.no_grad():
        prediction, _, _ = model(*inputs)
    if prediction.shape != (batch_size,):
        raise RuntimeError(
            f"{name}: expected output shape {(batch_size,)}, "
            f"got {tuple(prediction.shape)}"
        )
    print(f"PASS {name}: output shape {tuple(prediction.shape)}")


def make_graph_batch() -> Data:
    return Data(
        x=torch.rand(6, 78),
        edge_index=torch.tensor(
            [
                [0, 1, 1, 2, 3, 4, 4, 5],
                [1, 0, 2, 1, 4, 3, 5, 4],
            ],
            dtype=torch.long,
        ),
        batch=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        target=torch.randint(0, 26, (2, 1000), dtype=torch.long),
    )


def check_graph_model(name: str, model: torch.nn.Module) -> None:
    model.eval()
    with torch.no_grad():
        prediction = model(make_graph_batch())
    if prediction.shape != (2, 1):
        raise RuntimeError(
            f"{name}: expected output shape {(2, 1)}, "
            f"got {tuple(prediction.shape)}"
        )
    print(f"PASS {name}: output shape {tuple(prediction.shape)}")


def main() -> None:
    torch.manual_seed(42)

    check_vector_model("MLP", BaselineMLP())

    transformer_module.device = torch.device("cpu")
    check_vector_model(
        "Transformer",
        transformer_module.ReptileTransformer(),
    )
    check_vector_model(
        "Reptile-Transformer",
        transformer_module.ReptileTransformer(),
    )

    check_graph_model("GCN", GCNNet())
    check_graph_model("GAT", GATNet())
    check_graph_model("GIN", GINConvNet())
    check_graph_model("GAT-GCN", GAT_GCN())

    print("All 7 model entry points passed the synthetic forward test.")


if __name__ == "__main__":
    main()

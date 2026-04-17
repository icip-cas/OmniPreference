import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def normalize_hidden(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return X / (X.norm(p=2, dim=-1, keepdim=True) + eps)

def load_targets(obj):
    y = obj["y_softmax"]
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    return y.float()

@torch.no_grad()
def eval_one_layer_acc(
    X: torch.Tensor,
    y: torch.Tensor,
    clf: nn.Module,
    device: torch.device,
    batch_size: int,
) -> float:
    pin = (device.type == "cuda")
    dl = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=pin,
    )

    clf.eval()

    correct, total = 0, 0
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float()

        logits = clf(xb)                       
        pred = torch.argmax(logits, dim=-1)    

        gold = torch.argmax(yb, dim=-1)        

        correct += (pred == gold).sum().item()
        total += xb.size(0)

    return correct / max(1, total)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_obj = torch.load(args.test_pt, map_location="cpu")
    y_test = load_targets(test_obj)

    layer_keys = sorted([k for k in test_obj.keys() if isinstance(k, int)])

    layers = []
    accs = []

    for layer in layer_keys:
        ckpt_path = os.path.join(args.probe_dir, f"layer_{layer}.pt")
        if not os.path.isfile(ckpt_path):
            continue

        X = test_obj[layer]["h"].float()
        X = normalize_hidden(X)

        D = X.shape[1]
        clf = nn.Linear(D, 3, bias=True).to(device)
        clf.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        acc = eval_one_layer_acc(
            X=X,
            y=y_test,
            clf=clf,
            device=device,
            batch_size=args.batch_size,
        )

        print(f"layer={layer}\ttest_acc={acc:.6f}")
        layers.append(layer)
        accs.append(acc)

    os.makedirs(os.path.dirname(args.output_png) or ".", exist_ok=True)
    plt.figure()
    plt.plot(layers, accs, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Test Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_pt",
        type=str,
        default="hiddenstates/test.pt"
    )
    parser.add_argument(
        "--probe_dir",
        type=str,
        default="probe/"
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="test_acc_by_layer.png"
    )

    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    main(args)

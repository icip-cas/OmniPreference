import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def normalize_hidden(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
 
    return X / (X.norm(p=2, dim=-1, keepdim=True) + eps)

def soft_cross_entropy(preds, targets):
    log_probs = torch.log_softmax(preds, dim=-1)
    loss = -(targets * log_probs).sum(dim=-1).mean()
    return loss

def train_one_layer(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
):
    D = X_train.shape[1]
    clf = nn.Linear(D, 3, bias=True).to(device)
    
    opt = torch.optim.Adam(clf.parameters(), lr=lr)

    pin = (device.type == "cuda")
    train_dl = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=pin,
    )
    val_dl = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=pin,
    )

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        clf.train()
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            preds  = clf(xb)
            loss = soft_cross_entropy(preds, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        clf.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                preds = clf(xb)
                loss = soft_cross_entropy(preds, yb)


                bsz = xb.size(0)
                val_loss_sum += loss.item() * bsz
                val_n += bsz

        val_loss = val_loss_sum / max(1, val_n)

        print(f"Epoch {epoch+1}/{epochs} | val_SoftCE={val_loss:.6f}")
        
    
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in clf.state_dict().items()
            }
    clf.load_state_dict(best_state)
    return clf


def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_obj = torch.load(args.train_pt, map_location="cpu")
    val_obj = torch.load(args.val_pt, map_location="cpu")

    y_train = train_obj["y_softmax"].float()
    y_val = val_obj["y_softmax"].float()
    print(f"Label shape: {y_train.shape}, Label type: {y_train.dtype}")
    layer_keys = sorted([k for k in train_obj.keys() if isinstance(k, int)])

    os.makedirs(args.output_dir, exist_ok=True)

    for layer in layer_keys:
        print('layer:',layer)
        X_train = train_obj[layer]["h"].float()  # (N, D)
        X_val = val_obj[layer]["h"].float()      # (N, D)
  
        X_train = normalize_hidden(X_train)
        X_val = normalize_hidden(X_val)

        clf = train_one_layer(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

        torch.save(clf.state_dict(), os.path.join(args.output_dir, f"layer_{layer}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pt", type=str, default='train.pt')
    parser.add_argument("--val_pt", type=str, default='val.pt')
    parser.add_argument("--output_dir", type=str, default='probe_softmax')

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

import os
import json
import argparse
import torch
import torch.nn as nn


def normalize_hidden(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return X / (X.norm(p=2, dim=-1, keepdim=True) + eps)


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obj = torch.load(args.input_pt, map_location="cpu")

    model_outputs = obj["model_output"]

    all_results = {}

    for layer in range(args.start_layer, args.end_layer + 1):
        print(f"Processing layer {layer} ...")

        X = obj[layer]["h"].float()

        X = normalize_hidden(X)

        probe_path = os.path.join(args.probe_dir, f"layer_{layer}.pt")

        D = X.shape[1]
        clf = nn.Linear(D, 3, bias=True).to(device)

        clf.load_state_dict(torch.load(probe_path, map_location="cpu"))
        clf.eval()

        X = X.to(device)

        with torch.no_grad():
            logits = clf(X)
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        results = []

        for i in range(len(probs)):
            results.append({
                "id": i,
                "model_output": model_outputs[i],
                "pred": probs[i]
            })

        all_results[f"layer_{layer}"] = results

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {args.output_json}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_pt", type=str, default='Qwen2.5-Omni-7B-ahabench-hidden_states.pt')
    parser.add_argument("--probe_dir", type=str, default='probe')
    parser.add_argument("--output_json", type=str, default='pred_results/ahabench/aha_all_layers_result.json')
    parser.add_argument("--start_layer", type=int, default=1) 
    parser.add_argument("--end_layer", type=int, default=28)

    args = parser.parse_args()

    main(args)

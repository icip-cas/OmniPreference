"""Compute per-layer hallucination AUROC from distractor-modality scores."""

import argparse
import json
import os
import re

from sklearn.metrics import roc_auc_score


MODALITY_TO_INDEX = {
    "text": 0,
    "visual": 1,
    "audio": 2,
}


def is_hallucination(sample: dict) -> int:
    """Return 1 for hallucination and 0 for a correct model response."""
    answer_choice = {
        "yes": "A",
        "no": "B",
    }[sample["answer"].strip().lower()]

    model_choice = sample["model_raw_output"].strip()[0].upper()
    return int(model_choice != answer_choice)


def get_layer_numbers(all_layers: dict) -> list[int]:
    """Return numeric layer IDs from keys such as layer_1 and layer_28."""
    layer_numbers = []

    for layer_key in all_layers:
        match = re.fullmatch(r"layer_(\d+)", layer_key)
        if match:
            layer_numbers.append(int(match.group(1)))

    return sorted(layer_numbers)


def main(args: argparse.Namespace) -> None:
    with open(args.input_json, "r", encoding="utf-8") as file:
        all_layers = json.load(file)

    modality_index = MODALITY_TO_INDEX[args.distractor_modality]
    layer_numbers = get_layer_numbers(all_layers)
    layer_aurocs = []

    for layer in layer_numbers:
        probe_path = os.path.join(args.probe_dir, f"layer_{layer}.pt")
        if not os.path.isfile(probe_path):
            raise FileNotFoundError(f"Probe not found: {probe_path}")

        samples = all_layers[f"layer_{layer}"]

        hallucination_labels = [
            is_hallucination(sample)
            for sample in samples
        ]
        distractor_scores = [
            sample["pred"][modality_index]
            for sample in samples
        ]

        auroc = roc_auc_score(
            hallucination_labels,
            distractor_scores,
        )
        layer_aurocs.append((layer, auroc))

        print(f"Layer {layer}: AUROC = {auroc:.6f}")

    first_layer_samples = all_layers[f"layer_{layer_numbers[0]}"]
    first_layer_labels = [
        is_hallucination(sample)
        for sample in first_layer_samples
    ]
    hallucination_count = sum(first_layer_labels)
    correct_count = len(first_layer_labels) - hallucination_count

    output_dir = os.path.dirname(args.output_txt)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_txt, "w", encoding="utf-8") as file:
        file.write(f"Input JSON: {args.input_json}\n")
        file.write(f"Probe directory: {args.probe_dir}\n")
        file.write(f"Distractor modality: {args.distractor_modality}\n")
        file.write("Positive class: Hallucination\n")
        file.write(f"Correct samples: {correct_count}\n")
        file.write(f"Hallucination samples: {hallucination_count}\n")
        file.write("\n")

        for layer, auroc in layer_aurocs:
            file.write(f"layer_{layer}\tAUROC={auroc:.6f}\n")

    print(f"Saved AUROC results to {args.output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        default=(
            "01_Qwen2.5-Omni-7B-CMM-audio-driven_softmax_all_layers_result.json"
        ),
        help="Path to the all-layers Probe softmax result JSON.",
    )
    parser.add_argument(
        "--probe_dir",
        default='probe_softmax',
        help="Directory containing layer_1.pt, layer_2.pt, and so on.",
    )
    parser.add_argument(
        "--distractor_modality",
        choices=("text", "visual", "audio"),
        default='audio',
        help="Modality probability used as the hallucination detection score.",
    )
    parser.add_argument(
        "--output_txt",
       default='02_Qwen2.5-Omni-7B-CMM-audio-driven_AUROC_per_layer.txt',
        help="Path of the output TXT file.",
    )

    main(parser.parse_args())

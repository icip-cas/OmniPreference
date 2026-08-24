"""Compute the Mann-Whitney U test p-value reported in Table 2."""

import argparse
import json

from scipy.stats import mannwhitneyu


MODALITY_TO_INDEX = {
    "text": 0,
    "vision": 1,
    "audio": 2,
}


def is_correct(sample: dict) -> bool:
    answer_choice = {
        "yes": "A",
        "no": "B",
    }[sample["answer"].strip().lower()]

    model_choice = sample["model_raw_output"].strip()[0].upper()
    return model_choice == answer_choice


def split_distractor_probabilities(
    samples: list[dict],
    distractor_modality: str,
) -> tuple[list[float], list[float]]:
    distractor_index = MODALITY_TO_INDEX[distractor_modality]

    non_hallucination = []
    hallucination = []

    for sample in samples:
        probability = sample["pred"][distractor_index]

        if is_correct(sample):
            non_hallucination.append(probability)
        else:
            hallucination.append(probability)

    return non_hallucination, hallucination


def main(args: argparse.Namespace) -> None:
    with open(args.input_json, "r", encoding="utf-8") as file:
        all_layers = json.load(file)

    samples = all_layers[f"layer_{args.layer}"]

    non_hallucination, hallucination = split_distractor_probabilities(
        samples,
        args.distractor_modality,
    )

    _, p_value = mannwhitneyu(
        non_hallucination,
        hallucination,
        alternative="two-sided",
        method="asymptotic",
    )

    with open(args.output_txt, "w", encoding="utf-8") as file:
        file.write(f"p-value: {p_value:.12g}\n")

    print(f"p-value: {p_value:.12g}")
    print(f"Saved result to {args.output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Test whether distractor-modality preference probabilities differ "
            "between non-hallucination and hallucination samples."
        )
    )
    parser.add_argument(
        "--input_json",
        default=(
            "01_Qwen2.5-Omni-7B-CMM-"
            "language-driven_softmax_all_layers_result.json"
        ),
        help="All-layers softmax result JSON.",
    )
    parser.add_argument(
        "--distractor_modality",
        choices=("text", "vision", "audio"),
        default="text",
        help="Distractor modality defined for the evaluated task.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer containing the strongest modality-preference signal.",
    )
    parser.add_argument(
        "--output_txt",
        default=(
            "02_Qwen2.5-Omni-7B-CMM-language-driven_pvalue.txt"
        ),
        help="Output TXT file.",
    )

    main(parser.parse_args())
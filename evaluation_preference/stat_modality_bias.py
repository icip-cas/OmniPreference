import json
import re
import argparse
from collections import Counter
import os

import matplotlib.pyplot as plt


def parse_choice_from_response(response: str):
    if not isinstance(response, str):
        return None

    text = response.upper()

    match = re.search(r"\b([ABC])\b", text)
    if match:
        return match.group(1)

    return None


def stat_modality_counts(input_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modality_counter = Counter()
    total_samples = 0
    total_valid = 0
    invalid_parse = 0
    invalid_option = 0

    for sample in data:
        total_samples += 1

        raw = sample.get("model_raw_output")
        choice_letter = parse_choice_from_response(raw)

        if choice_letter is None:
            print(sample)
            invalid_parse += 1
            continue

        options = sample.get("options", [])
        chosen_modality = None
        for opt in options:
            if opt.get("option_id").upper() == choice_letter:
                chosen_modality = opt.get("modality")
                break

        if chosen_modality is None:
            invalid_option += 1
            continue

        modality_counter[chosen_modality] += 1
        total_valid += 1

    print(f"Total number of samples: {total_samples}")
    print(f"Number of valid samples (successfully parsed and matched): {total_valid}")
    print(
        f"Number of parsing failures "
        f"(model_raw_output could not be parsed as A/B/C): {invalid_parse}"
    )
    print(
        f"Number of matching failures "
        f"(parsed letter not found in options): {invalid_option}"
    )
    print()

    for modality in ["text", "image", "audio"]:
        count = modality_counter.get(modality, 0)
        ratio = count / total_samples if total_samples > 0 else 0.0
        print(f"{modality:>5s}: {count:5d}  ({ratio:.4f})")

    return modality_counter, total_valid, total_samples


def main():
    parser = argparse.ArgumentParser(
        description=(
            ""
        )
    )
    parser.add_argument(
        "--input",
        default=(
            "01_minicpm-o-2_6-promtp2-results.json"
        ),
        help=(
            "Path to the model output JSON file containing fields such as "
            "model_raw_output and options."
        ),
    )

    args = parser.parse_args()

    modality_counter, total_valid, total_samples = stat_modality_counts(args.input)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import re


MODALITIES = ("text", "image", "audio")
OPTION_PATTERN = re.compile(r"^\s*([ABC])(?:\s*[.:)\]]|\s|$)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Keep the conflict samples a model answers correctly when only one "
            "modality is presented."
        )
    )
    parser.add_argument(
        "--input",
        default="results.json",
        help="Unimodal inference result file.",
    )
    parser.add_argument(
        "--modality",
        choices=MODALITIES,
        default="text",
        help="Modality that was presented to the model.",
    )
    parser.add_argument(
        "--output",
        default="correct_results.json",
        help="File that receives the correctly answered samples.",
    )
    return parser.parse_args()


def get_selected_option(model_raw_output):
    if not isinstance(model_raw_output, str):
        return None
    match = OPTION_PATTERN.match(model_raw_output)
    if match is None:
        return None
    return match.group(1).upper()


def get_expected_option(sample, modality):
    for option in sample.get("options", []):
        if option.get("modality") == modality:
            return str(option.get("option_id")).strip().upper()
    return None


def write_json(path, samples):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(samples, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8-sig") as file:
        samples = json.load(file)
    if not isinstance(samples, list):
        raise TypeError(f"{args.input} must contain a JSON array")

    correct_samples = []
    unrecognized = 0
    missing_option = 0

    for sample in samples:
        selected = get_selected_option(sample.get("model_raw_output"))
        if selected is None:
            unrecognized += 1
            continue

        expected = get_expected_option(sample, args.modality)
        if expected is None:
            missing_option += 1
            continue

        if selected == expected:
            correct_samples.append(sample)

    write_json(args.output, correct_samples)

    total = len(samples)
    ratio = len(correct_samples) / total if total else 0.0
    print(f"Input samples: {total}")
    print(f"Presented modality: {args.modality}")
    print(f"Correct samples: {len(correct_samples)} ({ratio:.2%})")
    print(f"Samples without a {args.modality} option: {missing_option}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
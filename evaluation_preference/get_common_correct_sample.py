import argparse
import json
import os


EXCLUDED_FIELDS = ("model_raw_output",)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Intersect the correctly answered samples of all models and keep "
            "the entries whose fields are identical across every file."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["correct_results.json"],
        help="Correct-sample files, one per model.",
    )
    parser.add_argument(
        "--output",
        default="conflict_sample_1000.json",
        help="File that receives the shared samples.",
    )
    return parser.parse_args()


def identity_of(sample):
    payload = {
        key: value
        for key, value in sample.items()
        if key not in EXCLUDED_FIELDS
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_samples(path):
    with open(path, "r", encoding="utf-8-sig") as file:
        samples = json.load(file)
    if not isinstance(samples, list):
        raise TypeError(f"{path} must contain a JSON array")
    return samples


def index_by_identity(samples, path):
    indexed = {}
    for sample in samples:
        key = identity_of(sample)
        if key not in indexed:
            indexed[key] = sample
    if len(indexed) != len(samples):
        print(f"{path}: ignored {len(samples) - len(indexed)} duplicate entries")
    return indexed


def strip_excluded(sample):
    return {
        key: value
        for key, value in sample.items()
        if key not in EXCLUDED_FIELDS
    }


def write_json(path, samples):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(samples, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main():
    args = parse_args()

    shared_keys = None
    first_indexed = None

    for path in args.inputs:
        samples = load_samples(path)
        indexed = index_by_identity(samples, path)
        print(f"{path}: {len(samples)} samples, {len(indexed)} unique")

        if shared_keys is None:
            shared_keys = set(indexed)
            first_indexed = indexed
        else:
            shared_keys &= set(indexed)

    if shared_keys is None or first_indexed is None:
        raise ValueError("at least one input file is required")

    common_samples = [
        strip_excluded(sample)
        for key, sample in first_indexed.items()
        if key in shared_keys
    ]

    write_json(args.output, common_samples)

    print(f"Input files: {len(args.inputs)}")
    print(f"Common samples: {len(common_samples)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
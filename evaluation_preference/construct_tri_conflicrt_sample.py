from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


QUESTION = "Which option best describes what this example is mainly about?"
DEFAULT_COUNT = 5_000
DEFAULT_SEED = 20260801
FIELDS = ("text", "image", "audio")
NUM_CATEGORIES = 6
NUM_CATEGORY_TRIPLETS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate three-modality semantic-conflict samples."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("xmodbench_modality_path.json"),
        help="Aligned source JSON containing id/text/image/audio fields.",
    )
    parser.add_argument(
        "--categories",
        type=Path,
        default=Path("evaluation_preference/data/category.txt"),
        help="Markdown category file defining the six major categories.",
    )
    parser.add_argument(
        "--text-labels",
        type=Path,
        default=Path("evaluation_preference/data/text_label.json"),
        help="JSON containing canonical labels and their mapping ids.",
    )
    parser.add_argument(
        "--processed-text-labels",
        type=Path,
        default=Path("evaluation_preference/data/text_label_processed.json"),
        help="JSON containing processed text, matched to --text-labels by id.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("modality_conflict_1000.json"),
    )
    parser.add_argument("--num-samples", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--include-question",
        action="store_true",
        help="Add the standardized question as a top-level field.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        data: Any = json.load(file)

    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON array")

    required = {"id", "text", "image", "audio"}
    records: list[dict[str, str]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"record {index} must contain {sorted(required)}")
        record = {key: item[key] for key in required}
        if not all(isinstance(value, str) and value for value in record.values()):
            raise ValueError(f"record {index} has an empty or non-string field")
        records.append(record)

    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("source record ids must be unique")
    return records


def load_label_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data: Any = json.load(file)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON array")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or "id" not in item or "text" not in item:
            raise ValueError(f"{path}: record {index} must contain id and text")
        if not isinstance(item["text"], str) or not item["text"].strip():
            raise ValueError(f"{path}: record {index} has invalid text")
        try:
            hash(item["id"])
        except TypeError as error:
            raise ValueError(f"{path}: record {index} has an unhashable id") from error
        records.append({"id": item["id"], "text": item["text"]})
    return records


def load_processed_text_mapping(
    raw_path: Path,
    processed_path: Path,
    expected_labels: set[str],
) -> dict[str, str]:
    raw_records = load_label_file(raw_path)
    processed_records = load_label_file(processed_path)

    raw_by_id = {record["id"]: record["text"] for record in raw_records}
    processed_by_id = {
        record["id"]: record["text"] for record in processed_records
    }
    if len(raw_by_id) != len(raw_records):
        raise ValueError(f"{raw_path} contains duplicate ids")
    if len(processed_by_id) != len(processed_records):
        raise ValueError(f"{processed_path} contains duplicate ids")
    if set(raw_by_id) != set(processed_by_id):
        missing_processed = sorted(set(raw_by_id) - set(processed_by_id), key=str)
        missing_raw = sorted(set(processed_by_id) - set(raw_by_id), key=str)
        raise ValueError(
            "raw and processed label ids differ; "
            f"missing processed ids={missing_processed}, missing raw ids={missing_raw}"
        )

    mapping: dict[str, str] = {}
    for label_id, raw_text in raw_by_id.items():
        if raw_text in mapping:
            raise ValueError(f"{raw_path} contains duplicate label {raw_text!r}")
        mapping[raw_text] = processed_by_id[label_id]

    raw_labels = set(mapping)
    if raw_labels != expected_labels:
        missing = sorted(expected_labels - raw_labels)
        extra = sorted(raw_labels - expected_labels)
        raise ValueError(
            "text-label files do not exactly cover modality_path labels; "
            f"missing={missing}, extra={extra}"
        )
    return mapping


def parse_label_categories(markdown_path: Path, labels: Iterable[str]) -> dict[str, int]:
    markdown = markdown_path.read_text(encoding="utf-8")
    headings = [
        (match.start(), int(match.group(1)))
        for match in re.finditer(r"(?m)^\s*(\d+)\s*\.\s*[^\r\n]+", markdown)
    ]
    if not headings:
        raise ValueError(f"no numbered category headings found in {markdown_path}")

    mapping: dict[str, int] = {}
    for label in labels:
        categories: set[int] = set()
        for match in re.finditer(re.escape(label), markdown):
            preceding = [heading for heading in headings if heading[0] <= match.start()]
            if preceding:
                categories.add(preceding[-1][1])
        if len(categories) != 1:
            raise ValueError(
                f"label {label!r} maps to {len(categories)} Markdown categories: "
                f"{sorted(categories)}"
            )
        mapping[label] = next(iter(categories))
    return mapping


def balanced_allocation(items: list[Any], count: int, rng: random.Random) -> Counter:
    if not items:
        raise ValueError("cannot allocate over an empty collection")
    quotient, remainder = divmod(count, len(items))
    quotas: Counter = Counter({item: quotient for item in items})
    for item in rng.sample(items, remainder):
        quotas[item] += 1
    return quotas


def enumerate_category_triplets(category_ids: list[int]) -> list[tuple[int, int, int]]:
    triplets = list(itertools.combinations(sorted(category_ids), 3))
    if len(triplets) != NUM_CATEGORY_TRIPLETS:
        raise AssertionError(
            f"{NUM_CATEGORIES} categories must yield exactly "
            f"{NUM_CATEGORY_TRIPLETS} category triplets"
        )
    return triplets


def build_category_assignments(
    category_ids: list[int], count: int, seed: int
) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    triplets = enumerate_category_triplets(category_ids)
    triplet_quotas = balanced_allocation(triplets, count, rng)

    assignments: list[tuple[int, int, int]] = []
    for triplet in triplets:
        orders = list(itertools.permutations(triplet))
        order_quotas = balanced_allocation(orders, triplet_quotas[triplet], rng)
        for order in orders:
            assignments.extend([order] * order_quotas[order])
    rng.shuffle(assignments)
    return assignments


def balanced_option_orders(count: int, rng: random.Random) -> list[tuple[str, ...]]:
    permutations = list(itertools.permutations(FIELDS))
    rng.shuffle(permutations)
    orders = permutations * (count // len(permutations))
    orders.extend(permutations[: count % len(permutations)])
    rng.shuffle(orders)
    return orders


def make_label_picker(
    labels_by_category: dict[int, list[str]], rng: random.Random
) -> Callable[[int], str]:
    queues: dict[int, list[str]] = defaultdict(list)

    def pick(category: int) -> str:
        if not queues[category]:
            queues[category] = list(labels_by_category[category])
            rng.shuffle(queues[category])
        return queues[category].pop()

    return pick


def make_record_picker(
    records_by_label: dict[str, list[dict[str, str]]], rng: random.Random
) -> Callable[[str], dict[str, str]]:
    queues: dict[str, list[dict[str, str]]] = defaultdict(list)

    def pick(label: str) -> dict[str, str]:
        if not queues[label]:
            queues[label] = list(records_by_label[label])
            rng.shuffle(queues[label])
        return queues[label].pop()

    return pick


def build_samples(
    records: list[dict[str, str]],
    label_to_category: dict[str, int],
    processed_text: dict[str, str],
    count: int,
    seed: int,
    include_question: bool,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("--num-samples must be positive")
    if count > 1_000_000:
        raise ValueError("at most 1,000,000 six-digit conflict ids are available")

    records_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        records_by_label[record["text"]].append(record)

    labels_by_category: dict[int, list[str]] = defaultdict(list)
    for label in sorted(records_by_label):
        labels_by_category[label_to_category[label]].append(label)

    category_ids = sorted(labels_by_category)
    if len(category_ids) != NUM_CATEGORIES:
        raise ValueError(
            f"the category file must define exactly {NUM_CATEGORIES} categories, "
            f"found {len(category_ids)}"
        )

    assignments = build_category_assignments(category_ids, count, seed)
    option_orders = balanced_option_orders(count, random.Random(seed + 1))
    numeric_ids = random.Random(seed + 2).sample(range(1_000_000), count)
    label_picker = make_label_picker(labels_by_category, random.Random(seed + 3))
    record_picker = make_record_picker(records_by_label, random.Random(seed + 4))

    used_source_triples: set[tuple[str, str, str]] = set()
    samples: list[dict[str, Any]] = []

    for index, assignment in enumerate(assignments):
        labels_by_field: dict[str, str] = {}
        sources: dict[str, dict[str, str]] = {}
        for _ in range(200):
            labels_by_field = {
                field: label_picker(category)
                for field, category in zip(FIELDS, assignment)
            }
            sources = {
                field: record_picker(labels_by_field[field]) for field in FIELDS
            }
            source_triple = tuple(sources[field]["id"] for field in FIELDS)
            if source_triple not in used_source_triples:
                used_source_triples.add(source_triple)
                break
        else:
            raise RuntimeError("could not draw a unique conflict triplet")

        options = [
            {
                "option_id": chr(ord("A") + option_index),
                "label": labels_by_field[field],
                "modality": field,
            }
            for option_index, field in enumerate(option_orders[index])
        ]

        sample: dict[str, Any] = {
            "id": f"conflict_{numeric_ids[index]:06d}",
            "text": processed_text[labels_by_field["text"]],
            "image": sources["image"]["image"],
            "audio": sources["audio"]["audio"],
            "source_ids": {field: sources[field]["id"] for field in FIELDS},
            "labels": labels_by_field,
            "options": options,
        }
        if include_question:
            sample["question"] = QUESTION
        samples.append(sample)
    return samples


def validate_samples(
    samples: list[dict[str, Any]],
    records: list[dict[str, str]],
    label_to_category: dict[str, int],
    processed_text: dict[str, str],
    include_question: bool,
) -> None:
    source_by_id = {record["id"]: record for record in records}
    if len({sample["id"] for sample in samples}) != len(samples):
        raise AssertionError("generated ids are not unique")

    source_triples: set[tuple[str, str, str]] = set()
    order_counts: Counter[tuple[str, str, str]] = Counter()
    triplet_counts: Counter[tuple[int, int, int]] = Counter()
    assignment_counts: Counter[tuple[int, int, int]] = Counter()

    for sample in samples:
        if re.fullmatch(r"conflict_\d{6}", sample["id"]) is None:
            raise AssertionError(f"invalid sample id: {sample['id']}")

        labels = sample["labels"]
        if tuple(labels) != FIELDS or len(set(labels.values())) != 3:
            raise AssertionError("every sample must have three different labels")
        if not all(label in label_to_category for label in labels.values()):
            raise AssertionError("an output label is absent from the category file")

        categories = {field: label_to_category[labels[field]] for field in FIELDS}
        if len(set(categories.values())) != 3:
            raise AssertionError("every sample must draw from three distinct categories")

        source_ids = sample["source_ids"]
        if tuple(source_ids) != FIELDS or len(set(source_ids.values())) != 3:
            raise AssertionError("every sample must use three different source records")
        for field in FIELDS:
            source = source_by_id.get(source_ids[field])
            if source is None or source["text"] != labels[field]:
                raise AssertionError(f"{field} label does not match its source record")

        if sample["text"] != processed_text[labels["text"]]:
            raise AssertionError("top-level text is not the mapped processed text")
        if sample["image"] != source_by_id[source_ids["image"]]["image"]:
            raise AssertionError("image path does not match the image source id")
        if sample["audio"] != source_by_id[source_ids["audio"]]["audio"]:
            raise AssertionError("audio path does not match the audio source id")

        options = sample["options"]
        if [option["option_id"] for option in options] != ["A", "B", "C"]:
            raise AssertionError("option ids must be A, B, C")
        option_modalities = tuple(option["modality"] for option in options)
        if set(option_modalities) != set(FIELDS):
            raise AssertionError("options must cover text, image, and audio")
        for option in options:
            if option["label"] != labels[option["modality"]]:
                raise AssertionError("an option label does not match its modality")
        order_counts[option_modalities] += 1

        assignment = tuple(categories[field] for field in FIELDS)
        assignment_counts[assignment] += 1
        triplet_counts[tuple(sorted(assignment))] += 1
        source_triples.add(tuple(source_ids[field] for field in FIELDS))
        if include_question and sample.get("question") != QUESTION:
            raise AssertionError("the standardized question is missing or changed")

    if len(source_triples) != len(samples):
        raise AssertionError("generated triples must be unique")
    if len(triplet_counts) != NUM_CATEGORY_TRIPLETS:
        raise AssertionError(
            f"samples must cover all {NUM_CATEGORY_TRIPLETS} category triplets"
        )
    if max(triplet_counts.values()) - min(triplet_counts.values()) > 1:
        raise AssertionError("category triplets are not balanced")
    for triplet in triplet_counts:
        orders = list(itertools.permutations(triplet))
        counts = [assignment_counts[order] for order in orders]
        if max(counts) - min(counts) > 1:
            raise AssertionError(
                f"modality assignments within triplet {triplet} are not balanced"
            )
    if max(order_counts.values()) - min(order_counts.values()) > 1:
        raise AssertionError("the six option orders are not balanced")


def write_json(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(samples, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    labels = {record["text"] for record in records}
    label_to_category = parse_label_categories(args.categories, labels)
    processed_text = load_processed_text_mapping(
        args.text_labels,
        args.processed_text_labels,
        labels,
    )
    samples = build_samples(
        records,
        label_to_category,
        processed_text,
        args.num_samples,
        args.seed,
        args.include_question,
    )
    validate_samples(
        samples,
        records,
        label_to_category,
        processed_text,
        args.include_question,
    )
    write_json(args.output, samples)

    category_counts = Counter(label_to_category.values())
    triplet_counts = Counter(
        tuple(sorted(label_to_category[sample["labels"][field]] for field in FIELDS))
        for sample in samples
    )
    option_counts = Counter(
        tuple(option["modality"] for option in sample["options"])
        for sample in samples
    )
    print(f"source records: {len(records)}")
    print(f"raw/processed text mappings: {len(processed_text)}")
    print(f"unique labels: {len(labels)}; categories: {dict(sorted(category_counts.items()))}")
    print(f"category triplets: {len(triplet_counts)}")
    print(f"samples per category triplet: {dict(sorted(triplet_counts.items()))}")
    print(f"generated samples: {len(samples)}")
    print(f"question: {QUESTION}")
    print(f"question serialized: {args.include_question}")
    print(f"option-order counts: {dict(sorted(option_counts.items()))}")
    print(f"output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
import json
import re
import unicodedata
from pathlib import Path

INPUT_FILE = Path(
    r"casual_analysis/results/omnibench/minicpm-o-2_6-image-omnibench-(-0.7).json"
)


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    return "".join(
        char
        for char in text
        if not char.isspace()
        and unicodedata.category(char)[0] not in {"P", "S"}
    )


def extract_choice(model_raw_output):
    text = unicodedata.normalize(
        "NFKC", str(model_raw_output or "")
    ).strip()
    match = re.match(
        r"^\s*[\(\[]?([A-D])[\)\].:\-\s]",
        text,
        flags=re.I,
    )
    if match:
        return match.group(1).upper()
    matches = re.findall(
        r"(?:correct\s+answer\s+is|answer\s*(?:is|:))"
        r"\s*:?\s*[\(\[]?([A-D])(?:[\)\].:\-\s]|$)",
        text,
        flags=re.I,
    )
    if matches:
        return matches[-1].upper()

    matches = re.findall(
        r"(?:^|\n)\s*[\(\[]?([A-D])[\)\].:\-](?=\s|$)",
        text,
        flags=re.I,
    )
    return matches[-1].upper() if matches else None


with INPUT_FILE.open("r", encoding="utf-8") as file:
    samples = json.load(file)

correct = 0
unresolved = 0

for sample in samples:
    choice = extract_choice(sample.get("model_raw_output"))
    options = sample["options"]

    if choice is None:
        selected_option = None
        is_correct = False
        unresolved += 1
    else:
        choice_index = ord(choice) - ord("A")
        selected_option = (
            options[choice_index]
            if 0 <= choice_index < len(options)
            else None
        )
        is_correct = (
            selected_option is not None
            and normalize(selected_option) == normalize(sample["answer"])
        )

    sample["model_choice"] = choice
    sample["selected_option"] = selected_option
    sample["is_correct"] = is_correct

    correct += int(is_correct)

total = len(samples)
incorrect = total - correct
acc = correct / total if total else 0.0

print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"ACC: {correct}/{total} = {acc:.4%}")
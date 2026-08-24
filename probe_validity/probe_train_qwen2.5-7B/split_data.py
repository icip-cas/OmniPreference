import json
import random


INPUT_JSON = "Qwen2.5-Omni-7B-conflict_triplets-results.json"
TRAIN_JSON = "train.json"
VAL_JSON = "val.json"
TEST_JSON = "test.json"

TARGET_NUM = 1000

TRAIN_NUM = 800
VAL_NUM = 100
TEST_NUM = 100

random.seed(42)


def get_followed_modality(sample):
    output_id = str(sample.get("model_raw_output")).strip()
    options = sample.get("options")

    for opt in options:
        if str(opt.get("option_id")).strip() == output_id:
            return opt.get("modality")
    return None


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped = {
        "text": [],
        "image": [],
        "audio": []
    }

    for sample in data:
        modality = get_followed_modality(sample)
        if modality in grouped:
            grouped[modality].append(sample)

    train_data = []
    val_data = []
    test_data = []

    for modality in ["text", "image", "audio"]:
        selected = random.sample(grouped[modality], TARGET_NUM)
        random.shuffle(selected)

        train_data.extend(selected[:TRAIN_NUM])
        val_data.extend(selected[TRAIN_NUM:TRAIN_NUM + VAL_NUM])
        test_data.extend(selected[TRAIN_NUM + VAL_NUM:TRAIN_NUM + VAL_NUM + TEST_NUM])

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    with open(TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(VAL_JSON, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)

    with open(TEST_JSON, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    print(f"Saved: {TRAIN_JSON}, {VAL_JSON}, {TEST_JSON}")
    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")


if __name__ == "__main__":
    main()
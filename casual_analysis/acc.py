import json
import re

def extract_choice(text):
  
    m = re.search(r'\b([AB])\b', text)
    return m.group(1) if m else None

def main(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mapping = {'A': 'yes', 'B': 'no'}
    correct = 0
    total = 0

    for item in data:
        total += 1
        choice = extract_choice(item['model_raw_output'])
        if choice is None:
            continue
        pred = mapping[choice]
        gt = item['label'].strip().lower()
        if pred == gt:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f" acc: {accuracy:.4f}")

if __name__ == '__main__':
    main('minicpm-o-2_6-audio-driven-(0.7).json')   

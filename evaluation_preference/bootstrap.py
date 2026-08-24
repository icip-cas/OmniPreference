import json
import numpy as np

PATH = "Qwen3-Omni-30B-A3B-Instruct-results.json"
data = json.load(open(PATH, encoding="utf-8"))

results = []
invalid_ids = []

for d in data:
    id2mod = {o["option_id"]: o["modality"] for o in d["options"]}

    choice = d["model_raw_output"]
    if isinstance(choice, str):
        choice = choice.strip()

    if choice in id2mod:
        results.append(id2mod[choice])
    else:
        results.append(None)
        invalid_ids.append(d.get("id"))

results = np.array(results)

if invalid_ids:
    print(f"invalid {len(invalid_ids)} / {len(data)} = {len(invalid_ids)/len(data)*100:.2f}%")
    print(" id:")
    for sid in invalid_ids:
        print(f"    {sid}")
    print()


def bootstrap_ci(results, target, num_bootstraps=10000, ci=95):
    n = len(results)
    hit = (results == target).astype(float)
    idx = np.random.randint(0, n, size=(num_bootstraps, n))
    vals = hit[idx].mean(axis=1)
    lo = (100 - ci) / 2.0
    return vals.mean(), np.percentile(vals, lo), np.percentile(vals, 100 - lo)


np.random.seed(42)
n = len(results)
valid_n = n - len(invalid_ids)
print(f"Total samples N = {n} (valid samples {valid_n})\n")
for m in ["text", "image", "audio"]:
    point = np.mean(results == m)
    mean, lo, hi = bootstrap_ci(results, m)
    print(f"MSR({m:5s}): point est. {point*100:5.1f}%  |  95%CI [{lo*100:5.2f}%, {hi*100:5.2f}%]")

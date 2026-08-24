import os
import json
import base64
import argparse
import mimetypes
from tqdm import tqdm
from openai import OpenAI


def image_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def audio_to_base64(audio_path):
    ext = os.path.splitext(audio_path)[1].lower().lstrip(".") or "wav"
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, ext


def build_messages(sample):
    opts = {o["option_id"]: o["label"] for o in sample["options"]}
    letters = sorted(opts.keys())
    options_text = "\n".join(f"{l}. {opts[l]}" for l in letters)
    letter_choices = " or ".join(letters)

    question = (
        "which option best describes what this example is mainly about?\n"
        f"{options_text}\n"
        f"You should only output the single letter of your choice (A, B, or C), with no explanation or additional text."
    )

    image_url = image_to_data_url(sample["image"])
    audio_b64, audio_fmt = audio_to_base64(sample["audio"])

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample["text"]},
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": audio_fmt}},
                {"type": "text", "text": question},
            ],
        }
    ]


def run(args):
    with open(args.data_file, "r", encoding="utf-8") as f:
        conflict_data = json.load(f)
    print(f"Loaded {len(conflict_data)} conflict triplets from {args.data_file}")

    api_key = args.api_key
    if not api_key:
        raise ValueError("API key not provided. Use --api_key or set OPENROUTER_API_KEY env var.")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    predictions = []
    for sample in tqdm(conflict_data):
        messages = build_messages(sample)
        try:
            resp = client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                #temperature=args.temperature,
                #max_tokens=args.max_new_tokens,
            )
            output = resp.choices[0].message.content
        except Exception as e:
            print(f"Error on sample id={sample.get('id')}: {e}")
            output = f"ERROR: {e}"

        print(output)

        predictions.append(
            {
                "id": sample["id"],
                "text": sample["text"],
                "image": sample["image"],
                "audio": sample["audio"],
                "source_ids": sample.get("source_ids"),
                "labels": sample.get("labels"),
                "options": sample["options"],
                "model_raw_output": output,
            }
        )

    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = args.model_name.replace("/", "_")
    output_file = os.path.join(args.output_dir, f"{model_tag}-conflict_triplets-results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(predictions)} predictions to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default='conflict_triplets1000_processed.json')
    parser.add_argument("--output_dir", default='results')
    parser.add_argument("--model_name", default="google/gemini-3.1-pro-preview")
    parser.add_argument("--api_key", default='')
    parser.add_argument("--max_new_tokens", type=int, default=8)
    args = parser.parse_args()
    run(args)

import os
import json
import argparse
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import librosa


def build_messages(sample):

    text_label = sample["text"]          
    image_path = sample["image"]
    audio_path = sample["audio"]
    options = sample["options"]          

    options_dict = {opt["option_id"]: opt["label"] for opt in options}
    optA = options_dict["A"]
    optB = options_dict["B"]
    optC = options_dict["C"]

    image = Image.open(image_path).convert("RGB")

    audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)

    question = (
    "which option best describes what this example is mainly about?\n"
    f"A. {optA}\n"
    f"B. {optB}\n"
    f"C. {optC}\n"
    "You should only output the single letter of your choice (A, B, or C), with no explanation or additional text."
    )

    msgs = [
        {
            "role": "user",
            "content": [
                f"{text_label}", 
                image,                    
                audio_array,              
                question,                
            ],
        }
    ]

    return msgs


def run(args):
    data_file = args.data_file
    with open(data_file, "r", encoding="utf-8") as f:
        conflict_data = json.load(f)
    total = len(conflict_data)
    print(f"Loaded {total} conflict triplets from {data_file}")


    torch.manual_seed(100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading MiniCPM-o-2_6 model from {args.model_path}")
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,  
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    predictions = []

    for sample in tqdm(conflict_data):
        msgs = build_messages(sample)


        answer = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=False,                 
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
        )
        print('answer',answer)
        predictions.append(
            {
                "id": sample.get("id"),
                "text": sample.get("text"),
                "image": sample.get("image"),
                "audio": sample.get("audio"),
                "labels": sample.get("labels"),
                "options": sample["options"],
                "model_raw_output": answer,
            }
        )


    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/"))
    output_file = os.path.join(
        args.output_dir,
        f"{model_name}-weak-conflict-audio-results.json",
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    print(f"Saved predictions to {output_file}")
    print(f"Total samples: {total}")
    print("data_file:", args.data_file)
    print("model_path:", args.model_path)
    print("output_dir:", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_file",
        default="conflict_weak_sample_audio.json",
        help="Path to conflict_triplets.json",
    )
    parser.add_argument(
        "--model_path",
        default="minicpm-o-2_6",
        help="MiniCPM-o-2_6 model path or HF repo id.",
    )
    parser.add_argument(
        "--output_dir",
        default="result",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
        help="Max new tokens to generate.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",   
        help="Attention implementation: 'sdpa' or 'flash_attention_2'.",
    )

    args = parser.parse_args()
    run(args)

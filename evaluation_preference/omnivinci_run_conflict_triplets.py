import json
import os
import argparse
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModel, AutoConfig


def build_messages(sample):

    text_label = sample["text"]          
    image_path = sample["image"]
    audio_path = sample["audio"]
    options = sample["options"]         

    options_dict = {opt["option_id"]: opt["label"] for opt in options}

    optA = options_dict["A"]
    optB = options_dict["B"]
    optC = options_dict["C"]


    question = (
    "which option best describes what this example is mainly about?\n"
    f"A. {optA}\n"
    f"B. {optB}\n"
    f"C. {optC}\n"
    "You should only output the single letter of your choice (A, B, or C), with no explanation or additional text."
    )

    messages = [
        {
            "role": "user",
            "content": [
              
                {
                    "type": "text",
                    "text": f"{text_label}",
                },
                
                {
                    "type": "image",
                    "image": image_path,
                },
             
                {
                    "type": "audio",
                    "audio": audio_path,
                },
              
                {
                    "type": "text",
                    "text": question,
                },
            ],
        },
    ]

    return messages


def run(args):
    
    data_file = args.data_file
    with open(data_file, "r", encoding="utf-8") as f:
        conflict_data = json.load(f)
    total = len(conflict_data)
    print(f"Loaded {total} conflict triplets from {data_file}")

  
    model_path = args.model_path
    print(f"Loading OmniVinci model from {model_path}")

  
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "max_length": 99999999, 
    }
    generation_config = model.default_generation_config
    generation_config.update(**generation_kwargs)

    
    load_audio_in_video = True
    model.config.load_audio_in_video = load_audio_in_video
    processor.config.load_audio_in_video = load_audio_in_video


    if args.audio_length != "-1":
       
        model.config.audio_chunk_length = args.audio_length
        processor.config.audio_chunk_length = args.audio_length

    predictions = []

  
    for sample in tqdm(conflict_data):
        messages = build_messages(sample)

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


        inputs = processor([text])
        inputs = inputs.to(device)

        output_ids = model.generate(
            input_ids=inputs.input_ids,
            media=getattr(inputs, "media", None),
            media_config=getattr(inputs, "media_config", None),
            generation_config=generation_config,
        )
        response = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        response = response[0]
        print('response',response)

        predictions.append(
            {
                "id": sample.get("id"),
                "text": sample.get("text"),
                "image": sample.get("image"),
                "audio": sample.get("audio"),
                "labels": sample.get("labels"),
                "options": sample["options"],
                "model_raw_output": response,
            }
        )


    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(model_path.rstrip("/"))
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
        default="omnivinci",
        help="Path to OmniVinci model.",
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
        "--audio_length",
        type=str,
        default="max_3600",
        help='Maximum audio length, e.g. "max_3600" (following official demo). Use "-1" to disable.',
    )

    args = parser.parse_args()
    run(args)

import json
import os
import re
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info  
import argparse
from tqdm import tqdm


def build_messages(sample):

    text_label = sample["text"]          
    image_path = sample["image"]
    audio_path = sample["audio"]
    options = sample["options"]          

    options_dict = {opt["option_id"]: opt["label"] for opt in options}

    optA = options_dict["A"]
    optB = options_dict["B"]
    optC = options_dict["C"]

    # prompt 
    question = (
    "which option best describes what this example is mainly about?\n\n"
    f"A. {optA}\n"
    f"B. {optB}\n"
    f"C. {optC}\n\n"
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

    if "Qwen2.5" in args.model_path:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    predictions = []

    for sample in tqdm(conflict_data):
        messages = build_messages(sample)
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(model.device).to(model.dtype)

        output = model.generate(
            **inputs,
            use_audio_in_video=False,
            return_audio=False,
            max_new_tokens=args.max_new_tokens,
        )

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = output[:, prompt_len:]
        response = processor.batch_decode(
            gen_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        predictions.append(
            {
                "id": sample["id"],
                "options": sample["options"],
                "model_raw_output": response,
            }
        )
       
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/"))
    output_file = os.path.join(
        args.output_dir, f"{model_name}-conflict_tri-modality-results.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    print(f"Saved predictions to {output_file}")
    print(f"Total samples: {total}")
    
    print('data_file',args.data_file)
    print('model_path',args.model_path)
    print('output_dir',args.output_dir)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_json",
        default= 'data/conflict_triplets_processed.json'
    )
    parser.add_argument(
        "--model_path",
        default="huggingface_model/Qwen2.5-Omni-7B"
    )
    parser.add_argument(
        "--output_dir",
        default="data/"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=5
    )

    args = parser.parse_args()
    run(args)

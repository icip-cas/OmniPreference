import json
import os
import argparse
import torch
from tqdm import tqdm

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info


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

    
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",                      
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.to("cuda:0")
    model.disable_talker() 
  
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)

    predictions = []

    for sample in tqdm(conflict_data):
        messages = build_messages(sample)

       
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        audios, images, videos = process_mm_info( messages, use_audio_in_video=False )

       
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        
        prompt_len = inputs["input_ids"].shape[1]
        inputs = inputs.to(model.device).to(model.dtype)

      
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                thinker_max_new_tokens=args.max_new_tokens,
                use_audio_in_video=False,
                return_audio=False,
                thinker_return_dict_in_generate=False,
                thinker_do_sample=False,
            )


        seq = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
        seq = seq.sequences if hasattr(seq, "sequences") else seq

        gen_ids = seq[:, prompt_len:] if seq.shape[1] > prompt_len else seq
        response = processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

     
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
    model_name = os.path.basename(args.model_path.rstrip("/"))
    output_file = os.path.join(
        args.output_dir, f"{model_name}-prompt1-results.json"
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
        "--data_file",
        default="evaluation_preference/data/conflict_sample_1000.json",
        help="Path to conflict_triplets.json",
    )
    parser.add_argument(
        "--model_path",
        default="Qwen3-Omni-30B-A3B-Instruct",
        help="The model to evaluate.",
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

    args = parser.parse_args()
    run(args)

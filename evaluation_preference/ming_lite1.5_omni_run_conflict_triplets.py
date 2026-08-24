import json
import os
import torch
import argparse
from tqdm import tqdm

from transformers import AutoProcessor, GenerationConfig
from modeling_bailingmm import BailingMMNativeForConditionalGeneration


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
            "role": "HUMAN",
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

    print(f"Loading Ming-Lite-Omni-1.5 model from: {args.model_path}")
    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,      
        #attn_implementation="flash_attention_2",
        load_image_gen=True,
        low_cpu_mem_usage=True,
    ).to(args.device)

    print(f"Loading processor from: {args.processor_path}")
    processor = AutoProcessor.from_pretrained(
        args.processor_path,
        trust_remote_code=True
    )

    generation_config = GenerationConfig.from_dict(
        {"no_repeat_ngram_size": args.no_repeat_ngram_size}
    )

    eos_token_id = getattr(processor, "gen_terminator", None)
    if eos_token_id is None:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and getattr(tok, "eos_token_id", None) is not None:
            eos_token_id = tok.eos_token_id

    predictions = []

    for sample in tqdm(conflict_data):
        messages = build_messages(sample)

        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
            audio_kwargs={"use_whisper_encoder": True},
        )
        inputs = inputs.to(model.device)

        for k in inputs.keys():
            if k in ["pixel_values", "pixel_values_videos", "audio_feats"]:
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
            "generation_config": generation_config,
        }
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id

        generated_ids = model.generate(
            **inputs,
            **generate_kwargs,
        )
        

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = generated_ids[:, prompt_len:]

        response = processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        print('response', response)

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
        args.output_dir, f"{model_name}-results.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    print(f"Saved predictions to {output_file}")
    print(f"Total samples: {total}")
    print("data_file:", args.data_file)
    print("model_path:", args.model_path)
    print("processor_path:", args.processor_path)
    print("output_dir:", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_file",
        default="evaluation_preference/data/conflict_sample_1000.json",
        help="Path to conflict_triplets.json",
    )
    parser.add_argument(
        "--model_path",
        default="Ming-Lite-Omni-1.5",
        help="The Ming-Lite-Omni-1.5 model path or HF repo id.",
    )
    parser.add_argument(
        "--processor_path",
        default=".",
        help="Path to load AutoProcessor (e.g., '.' or same as model_path).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run model on, e.g. 'cuda' or 'cpu'.",
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
        "--no_repeat_ngram_size",
        type=int,
        default=10,
        help="no_repeat_ngram_size for GenerationConfig.",
    )

    args = parser.parse_args()
    run(args)


import json
import os
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import argparse
from tqdm import tqdm


def build_messages(sample, video_dir):
    video_path = os.path.join(video_dir, f"{sample['video_id']}.mp4")
    question = sample["text"]

    prompt = (
        f"{question}\n"
        "A. Yes. B. No.\n"
        "Select the best option for the question."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    return messages


def run(args):
    with open(args.data_file, "r", encoding="utf-8") as f:
        conflict_data = json.load(f)

    print(f"Loaded {len(conflict_data)} samples from {args.data_file}")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model.eval()

    layer_to_h_list = None
    model_outputs = []
    predictions = []

    for sample in tqdm(conflict_data):
        messages = build_messages(sample, args.video_dir)

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=False,
            use_audio_in_video=True,
        )

        prompt_len = inputs["input_ids"].shape[1]
        last_prompt_pos = prompt_len - 1

        inputs = inputs.to(model.device).to(model.dtype)

        # ---- hidden state ----
        with torch.no_grad():
            thinker_out = model.thinker(
                **inputs,
                use_audio_in_video=True,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = thinker_out.hidden_states
            num_layers_total = len(hidden_states)

            if layer_to_h_list is None:
                layer_to_h_list = {l: [] for l in range(1, num_layers_total)}

            for l in range(1, num_layers_total):
                hs = hidden_states[l]               # (1, S, D)
                h_last = hs[:, last_prompt_pos, :]  # (1, D)
                layer_to_h_list[l].append(
                    h_last.squeeze(0).to(dtype=torch.float32, device="cpu")
                )

        # ---- generate ----
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=False,
                max_new_tokens=args.max_new_tokens,
            )

        gen_ids = gen[:, prompt_len:]
        response = processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        model_outputs.append(response)

        predictions.append(
            {
                "video_id": sample["video_id"],
                "task": sample["task"],
                "question": sample["text"],
                "answer": sample["label"],
                "model_raw_output": response,
            }
        )

    save_obj = {
        l: {"h": torch.stack(layer_to_h_list[l], dim=0)}
        for l in sorted(layer_to_h_list.keys())
    }
    save_obj["model_output"] = model_outputs

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/"))

    output_json = os.path.join(args.output_dir, f"{model_name}-AVH-video-driven-audio-results.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)
    print(f"Saved predictions to {output_json}")

    output_pt = os.path.join(
        args.output_dir,
        f"{model_name}-AVH-video-driven-audio-hiddenstates.pt"
    )
    torch.save(save_obj, output_pt)
    print(f"Saved hidden states to {output_pt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        default="avh-video-driven-audio-hallucination.json",
        help="Path to AVHBench json file",
    )
    parser.add_argument(
        "--model_path",
        default="Qwen2.5-Omni-7B",
        help="The model to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        default="result",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int, default=5,
        help="Max new tokens to generate.")
    parser.add_argument(
        "--video_dir",
        default="videos",
        help="Directory containing video files",
    )
    args = parser.parse_args()
    run(args)

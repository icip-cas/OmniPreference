

import argparse
import json
import os
from typing import Any

import torch
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)


def build_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a video-frames + text conversation."""
    content: list[dict[str, str]] = [
        {
            "type": "video",
            "video": sample["video_path"],
        }
    ]

    prompt = (
        f"{sample['question']}\n"
        "A. Yes. B. No.\n"
        "Select the best option for the question."
    )
    content.append({"type": "text", "text": prompt})

    return [{"role": "user", "content": content}]


def run(args: argparse.Namespace) -> None:
    with open(args.data_file, "r", encoding="utf-8") as file:
        samples = json.load(file)

    print(f"Loaded {len(samples)} samples from {args.data_file}")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model.eval()

    layer_to_hidden_states: dict[int, list[torch.Tensor]] | None = None
    model_outputs: list[str] = []
    sample_ids: list[str] = []
    predictions: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="CMM language-driven"):
        messages = build_messages(sample)

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        audios, images, videos = process_mm_info(
            messages,
            use_audio_in_video=False,
        )
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=False,
            use_audio_in_video=False,
        )

        prompt_len = inputs["input_ids"].shape[1]
        last_prompt_pos = prompt_len - 1
        inputs = inputs.to(model.device).to(model.dtype)

        with torch.inference_mode():
            thinker_output = model.thinker(
                **inputs,
                use_audio_in_video=False,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = thinker_output.hidden_states

        if layer_to_hidden_states is None:
            layer_to_hidden_states = {
                layer: [] for layer in range(1, len(hidden_states))
            }

        for layer in range(1, len(hidden_states)):
            hidden_last = hidden_states[layer][0, last_prompt_pos, :]
            layer_to_hidden_states[layer].append(
                hidden_last.detach().to(device="cpu", dtype=torch.float32)
            )

        del thinker_output, hidden_states

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                use_audio_in_video=False,
                return_audio=False,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        response_ids = generated_ids[:, prompt_len:]
        response = processor.batch_decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        sample_ids.append(sample["id"])
        model_outputs.append(response)
        prediction = dict(sample)
        prediction["model_raw_output"] = response
        predictions.append(prediction)

        del generated_ids, response_ids, inputs

    save_object: dict[Any, Any] = {
        layer: {"h": torch.stack(values, dim=0)}
        for layer, values in sorted(layer_to_hidden_states.items())
    }
    save_object["sample_ids"] = sample_ids
    save_object["model_output"] = model_outputs

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/\\"))

    output_json = os.path.join(
        args.output_dir, f"{model_name}-CMM-language-driven-results.json"
    )
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(predictions, file, indent=2, ensure_ascii=False)
    print(f"Saved predictions to {output_json}")

    output_pt = os.path.join(
        args.output_dir, f"{model_name}-CMM-language-driven-hidden-states.pt"
    )
    torch.save(save_object, output_pt)
    print(f"Saved hidden states to {output_pt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Qwen2.5-Omni responses and hidden states for CMM language-driven data."
    )
    parser.add_argument(
        "--data_file",
        default="cmm-language-driven.json",
        help="Path to cmm_language_driven.json.",
    )
    parser.add_argument(
        "--model_path",
        default="Qwen2.5-Omni-7B",
        help="Local Qwen2.5-Omni model path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--output_dir",
        default="result",
        help="Directory for the JSON and PT result files.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=5,
        help="Maximum number of response tokens to generate.",
    )
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        help="Transformers attention implementation (for example flash_attention_2 or sdpa).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

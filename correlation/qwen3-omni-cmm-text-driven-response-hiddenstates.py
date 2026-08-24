import json
import os
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import argparse
from tqdm import tqdm


def build_messages(sample):
    video_path = sample["video_path"]
    question = sample["question"]

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

    with open(args.data_file, "r", encoding="utf-8-sig") as f:
        conflict_data = json.load(f)

    print(f"Loaded {len(conflict_data)} samples from {args.data_file}")

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)
    model.eval()

    layer_to_h_list = None
    model_outputs = []
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
            padding=False,
            use_audio_in_video=False,
        )

        prompt_len = inputs["input_ids"].shape[1]

        inputs = inputs.to(model.device).to(model.dtype)

        with torch.no_grad():

            gen_out = model.thinker.generate(
                **inputs,
                use_audio_in_video=False,
                max_new_tokens=args.max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )


        prefill_hidden_states = gen_out.hidden_states[0]
        num_layers_total = len(prefill_hidden_states)

        if layer_to_h_list is None:
            layer_to_h_list = {l: [] for l in range(1, num_layers_total)}

        for l in range(1, num_layers_total):
            hs = prefill_hidden_states[l]
            h_last = hs[:, -1, :]
            layer_to_h_list[l].append(
                h_last.squeeze(0).to(dtype=torch.float32, device="cpu")
            )


        gen_ids = gen_out.sequences[:, prompt_len:]
        response = processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        model_outputs.append(response)

        predictions.append(
            {
                "id": sample["id"],
                "category": sample["category"],
                "sub_category": sample["sub_category"],
                "modality": sample["modality"],
                "granularity": sample["granularity"],
                "video_path": sample["video_path"],
                "question": sample["question"],
                "answer": sample["answer"],
                "model_raw_output": response,
            }
        )


        del gen_out, prefill_hidden_states
        torch.cuda.empty_cache()


    save_obj = {
        l: {"h": torch.stack(layer_to_h_list[l], dim=0)}
        for l in sorted(layer_to_h_list.keys())
    }
    save_obj["model_output"] = model_outputs

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/"))

    output_json = os.path.join(args.output_dir, f"{model_name}-CMM-language-driven-results.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)
    print(f"Saved predictions to {output_json}")

    output_pt = os.path.join(
        args.output_dir,
        f"{model_name}-CMM-language-driven-hiddenstates.pt"
    )
    torch.save(save_obj, output_pt)
    print(f"Saved hidden states to {output_pt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        default="cmm-language-driven.json",
        help="Path to CMM language-driven json file",
    )
    parser.add_argument(
        "--model_path",
        default="Qwen3-Omni-30B-A3B-Instruct",
        help="The model to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        default="correlation/result",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int, default=5,
        help="Max new tokens to generate.")
    args = parser.parse_args()
    run(args)

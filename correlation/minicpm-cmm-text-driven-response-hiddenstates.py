import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent


def load_video_frames(video_path: str, max_num_frames: int) -> list[Image.Image]:

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    video_reader = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(video_reader)
    if total_frames == 0:
        raise ValueError(f"Video contains no frames: {video_path}")
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        min(max_num_frames, total_frames),
        dtype=int,
    )
    frames = video_reader.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


def resolve_media_path(stored_path: str, data_file: str, cmm_root: str | None) -> str:

    if not stored_path:
        raise ValueError("The sample has an empty video path.")
    if cmm_root:
        parts = stored_path.replace("\\", "/").split("/")
        cmm_indices = [i for i, part in enumerate(parts) if part.lower() == "cmm"]
        if not cmm_indices:
            raise ValueError(
                f"Cannot remap path without a CMM component: {stored_path}"
            )
        return os.path.join(cmm_root, *parts[cmm_indices[-1] + 1 :])
    if os.path.isabs(stored_path):
        return stored_path
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(data_file)), stored_path)
    )


def build_messages(
    sample: dict,
    data_file: str,
    cmm_root: str | None,
    max_num_frames: int,
) -> list[dict]:

    video_path = resolve_media_path(sample["video_path"], data_file, cmm_root)
    frames = load_video_frames(video_path, max_num_frames)
    prompt = (
        f'{sample["question"]}\n'
        "A. Yes. B. No.\n"
        "Select the best option for the question."
    )
    return [{"role": "user", "content": frames + [prompt]}]


def prepare_inputs(model, tokenizer, messages: list[dict], max_length: int):

    images = []
    copied_messages = [
        {"role": message["role"], "content": list(message["content"])}
        for message in messages
    ]
    for message in copied_messages:
        content_parts = []
        for content in message["content"]:
            if isinstance(content, Image.Image):
                images.append(content)
                content_parts.append("(<image>./</image>)")
            elif isinstance(content, str):
                content_parts.append(content)
            else:
                raise TypeError(f"Unsupported message content: {type(content)!r}")
        message["content"] = "\n".join(content_parts)

    prompt_text = tokenizer.apply_chat_template(
        copied_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = model.processor(
        [prompt_text],
        [images],
        [[]],
        [[]],
        max_slice_nums=None,
        use_image_id=None,
        chunk_input=False,
        return_tensors="pt",
        max_length=max_length,
    ).to(model.device)
    inputs.pop("image_sizes", None)
    return inputs


def get_inputs_embeds(model, inputs):

    inputs_embeds, _ = model.get_vllm_embedding(inputs)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long
        ).unsqueeze(0)
    else:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    return inputs_embeds, position_ids, attention_mask


def decode_response(tokenizer, generated_ids: torch.Tensor) -> str:
    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    tts_end = getattr(tokenizer, "tts_end", None)
    if tts_end:
        response = response.replace(tts_end, "")
    return response.strip()


def run(args: argparse.Namespace) -> None:
    with open(args.data_file, "r", encoding="utf-8") as file:
        samples = json.load(file)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected a non-empty JSON list: {args.data_file}")
    if args.limit is not None:
        samples = samples[: args.limit]
    print(f"Loaded {len(samples)} samples from {args.data_file}")

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        init_vision=True,
        init_audio=False,
        init_tts=False,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    terminators = [
        tokenizer.convert_tokens_to_ids(token) for token in model.terminators
    ]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    layer_hidden_states = None
    model_outputs = []
    predictions = []
    for sample in tqdm(samples, desc="CMM language-driven"):
        messages = build_messages(
            sample,
            args.data_file,
            args.cmm_root,
            args.max_num_frames,
        )
        inputs = prepare_inputs(model, tokenizer, messages, args.max_length)

        with torch.inference_mode():
            inputs_embeds, position_ids, attention_mask = get_inputs_embeds(
                model, inputs
            )
            last_prompt_position = inputs_embeds.shape[1] - 1
            llm_output = model.llm(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = llm_output.hidden_states
            if layer_hidden_states is None:
                layer_hidden_states = {
                    layer: [] for layer in range(1, len(hidden_states))
                }
            for layer in layer_hidden_states:
                layer_hidden_states[layer].append(
                    hidden_states[layer][0, last_prompt_position]
                    .to(device="cpu", dtype=torch.float32)
                    .clone()
                )

            generation_output = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                eos_token_id=terminators,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )

        response = decode_response(tokenizer, generation_output.sequences)
        model_outputs.append(response)
        prediction = dict(sample)
        prediction["model_raw_output"] = response
        predictions.append(prediction)
        print(f'[{sample.get("id", len(predictions) - 1)}] {response}')

    hidden_state_output = {
        layer: {"h": torch.stack(values, dim=0)}
        for layer, values in sorted(layer_hidden_states.items())
    }
    hidden_state_output["model_output"] = model_outputs
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(os.path.normpath(args.model_path))
    json_path = os.path.join(
        args.output_dir, f"{model_name}-CMM-language-driven-results.json"
    )
    hidden_path = os.path.join(
        args.output_dir, f"{model_name}-CMM-language-driven-hiddenstates.pt"
    )
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(predictions, file, indent=4, ensure_ascii=False)
    torch.save(hidden_state_output, hidden_path)
    print(f"Saved predictions to {json_path}")
    print(f"Saved hidden states to {hidden_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_file", default="cmm-language-driven.json"
    )
    parser.add_argument(
        "--model_path",
        default="minicpm-o-2_6",
    )
    parser.add_argument(
        "--output_dir", default="correlation/result"
    )
    parser.add_argument(
        "--cmm_root",
        default=None,
        help="Optional local CMM root used to replace the CMM prefix in JSON paths.",
    )
    parser.add_argument("--max_num_frames", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N samples."
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

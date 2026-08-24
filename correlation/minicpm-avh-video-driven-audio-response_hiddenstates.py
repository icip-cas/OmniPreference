import json
import os
import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, AudioReader, cpu, bridge


def load_video_frames(video_path: str, max_num_frames: int = 64) -> list:

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    indices = np.linspace(0, total - 1, min(max_num_frames, total), dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(f).convert("RGB") for f in frames]


def load_audio_from_video(video_path: str, target_sr: int = 16000) -> np.ndarray:
    bridge.set_bridge("native")
    ar = AudioReader(video_path, ctx=cpu(0), sample_rate=target_sr, mono=True)
    wav = ar[:].asnumpy()
    wav = wav.squeeze(0)
    return wav.astype(np.float32)


def get_inputs_embeds(model, inputs):


    vllm_embedding, _ = model.get_vllm_embedding(inputs)


    if model.config.init_audio:
        vllm_embedding = model.get_omni_embedding(
            inputs,
            input_embeddings=vllm_embedding,
            chunk_length=model.config.audio_chunk_length,
        )


    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    else:
        seq_len = vllm_embedding.shape[1]
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=vllm_embedding.device
        ).unsqueeze(0)

    return vllm_embedding, position_ids, attention_mask


def build_msgs(sample, video_dir: str, max_num_frames: int = 64):

    video_path = os.path.join(video_dir, f"{sample['video_id']}.mp4")
    question   = sample["text"]

    prompt = (
        f"{question}\n"
        "A. Yes. B. No.\n"
        "Select the best option for the question."
    )

    frames = load_video_frames(video_path, max_num_frames=max_num_frames)
    audio  = load_audio_from_video(video_path, target_sr=16000)


    content = frames + [audio, prompt]

    msgs = [{"role": "user", "content": content}]
    return msgs


def run(args):

    with open(args.data_file, "r", encoding="utf-8") as f:
        conflict_data = json.load(f)

    print(f"Loaded {len(conflict_data)} samples from {args.data_file}")


    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        init_vision=True,
        init_audio=True,
        init_tts=False,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor  = model.processor

    terminators = [tokenizer.convert_tokens_to_ids(t) for t in model.terminators]


    layer_to_h_list = None
    model_outputs   = []
    predictions     = []


    for sample in tqdm(conflict_data):
        msgs = build_msgs(sample, args.video_dir, max_num_frames=args.max_num_frames)


        images_in_msg = []
        audios_in_msg = []
        audio_parts   = []
        copy_msgs     = [{"role": m["role"], "content": list(m["content"])} for m in msgs]

        for i, msg in enumerate(copy_msgs):
            cur_msgs = []
            for c in msg["content"]:
                if isinstance(c, Image.Image):
                    images_in_msg.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, np.ndarray):
                    audios_in_msg.append(c)
                    audio_parts.append(i)
                    cur_msgs.append("(<audio>./</audio>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)


        prompt_text = tokenizer.apply_chat_template(
            copy_msgs,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=model.default_tts_chat_template,
        )

        inputs = processor(
            [prompt_text],
            [images_in_msg],
            [audios_in_msg],
            [audio_parts],
            max_slice_nums=None,
            use_image_id=None,
            chunk_input=False,
            return_tensors="pt",
            max_length=32768,
        ).to(model.device)

        inputs.pop("image_sizes", None)


        with torch.no_grad():
            inputs_embeds, position_ids, attention_mask = get_inputs_embeds(model, inputs)

            prompt_len      = inputs_embeds.shape[1]
            last_prompt_pos = prompt_len - 1


            llm_out = model.llm(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states    = llm_out.hidden_states
            num_layers_total = len(hidden_states)

            if layer_to_h_list is None:
                layer_to_h_list = {l: [] for l in range(1, num_layers_total)}

            for l in range(1, num_layers_total):
                hs     = hidden_states[l]
                h_last = hs[:, last_prompt_pos, :]
                layer_to_h_list[l].append(
                    h_last.squeeze(0).to(dtype=torch.float32, device="cpu")
                )


            gen_out = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=0,
                eos_token_id=terminators,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )

        gen_ids  = gen_out.sequences
        response = tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        response = response.replace(tokenizer.tts_end, "").strip()

        model_outputs.append(response)
        predictions.append(
            {
                "video_id":         sample["video_id"],
                "task":             sample["task"],
                "question":         sample["text"],
                "answer":           sample["label"],
                "model_raw_output": response,
            }
        )
        print(response)


    save_obj = {
        l: {"h": torch.stack(layer_to_h_list[l], dim=0)}
        for l in sorted(layer_to_h_list.keys())
    }
    save_obj["model_output"] = model_outputs

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/"))

    output_json = os.path.join(
        args.output_dir, f"{model_name}-AVH-video-driven-audio-results.json"
    )
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)
    print(f"Saved predictions to {output_json}")

    output_pt = os.path.join(
        args.output_dir, f"{model_name}-AVH-video-driven-audio-hiddenstates.pt"
    )
    torch.save(save_obj, output_pt)
    print(f"Saved hidden states to {output_pt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        default="avh-video-driven-audio-hallucination.json",
    )
    parser.add_argument(
        "--model_path",
        default="minicpm-o-2_6",
    )
    parser.add_argument(
        "--output_dir",
        default="correlation/result",
    )
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument(
        "--video_dir",
        default="videos",
    )
    parser.add_argument("--max_num_frames", type=int, default=16)

    args = parser.parse_args()
    run(args)

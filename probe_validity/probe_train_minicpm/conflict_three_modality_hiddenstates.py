import json
import os
import argparse
import re
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def get_single_token_id(tokenizer, s: str) -> int:
    for cand in (s, " " + s, "\n" + s):
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"Cannot find single token id for '{s}'")


def option_id_by_modality(sample):
    mod_to_opt = {}
    for opt in sample["options"]:
        mod = opt["modality"]
        oid = opt["option_id"]
        mod_to_opt[mod] = oid

    text_oid = mod_to_opt.get("text")
    image_oid = mod_to_opt.get("image")
    audio_oid = mod_to_opt.get("audio")
    return text_oid, image_oid, audio_oid


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    wav, sr = librosa.load(path, sr=target_sr, mono=True)
    return wav.astype(np.float32)


def build_msgs(sample):
    text_label = sample["text"]
    image_path = sample["image"]
    audio_path = sample["audio"]
    options = sample["options"]

    options_dict = {opt["option_id"]: opt["label"] for opt in options}
    optA = options_dict["A"]
    optB = options_dict["B"]
    optC = options_dict["C"]

    question = (
        "which option best describes what this example is mainly about?\n\n"
        f"A. {optA}\n"
        f"B. {optB}\n"
        f"C. {optC}\n\n"
        "You should only output the single letter of your choice (A, B, or C), with no explanation or additional text."
    )

    image = Image.open(image_path).convert("RGB")
    audio = load_audio(audio_path, target_sr=16000)

    msgs = [
        {
            "role": "user",
            "content": [
                text_label,
                image,
                audio,
                question,
            ],
        }
    ]
    return msgs


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
        position_ids = torch.arange(seq_len, dtype=torch.long, device=vllm_embedding.device).unsqueeze(0)

    return vllm_embedding, position_ids, attention_mask


def run(args):
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

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
    processor = model.processor

    id_A = get_single_token_id(tokenizer, "A")
    id_B = get_single_token_id(tokenizer, "B")
    id_C = get_single_token_id(tokenizer, "C")
    optid_to_tokid = {"A": id_A, "B": id_B, "C": id_C}
    print("optid_to_tokid:", optid_to_tokid)

    layer_to_h_list = None
    y_softmax = []
    model_outputs = []

    terminators = [tokenizer.convert_tokens_to_ids(t) for t in model.terminators]

    for sample in tqdm(data):
        msgs = build_msgs(sample)

        images_in_msg = []
        audios_in_msg = []
        audio_parts = []
        copy_msgs = deepcopy(msgs)

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

        prompt = tokenizer.apply_chat_template(
            copy_msgs,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=model.default_tts_chat_template,
        )

        inputs = processor(
            [prompt],
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

            prompt_len = inputs_embeds.shape[1]
            last_prompt_pos = prompt_len - 1

            llm_out = model.llm(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = llm_out.hidden_states
            num_layers_total = len(hidden_states)

            if layer_to_h_list is None:
                layer_to_h_list = {l: [] for l in range(1, num_layers_total)}

            for l in range(1, num_layers_total):
                hs = hidden_states[l]
                h_last = hs[:, last_prompt_pos, :]
                layer_to_h_list[l].append(
                    h_last.squeeze(0).to(dtype=torch.float32, device="cpu")
                )

            text_oid, image_oid, audio_oid = option_id_by_modality(sample)
            tok_ids_in_mod_order = torch.tensor(
                [
                    optid_to_tokid[text_oid],
                    optid_to_tokid[image_oid],
                    optid_to_tokid[audio_oid],
                ],
                device=llm_out.logits.device,
            )

            next_token_logits = llm_out.logits[0, -1, :].to(torch.float32)

            probs_full = F.softmax(next_token_logits, dim=-1)
            probs_tia = probs_full[tok_ids_in_mod_order].to("cpu")
            y_softmax.append(probs_tia)

            gen_out = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=0,
                eos_token_id=terminators,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )

            gen_ids = gen_out.sequences[:, prompt_len:]
            response = tokenizer.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            response = response.replace(tokenizer.tts_end, "").strip()
            model_outputs.append(response)

    y_softmax = torch.stack(y_softmax, dim=0)

    save_obj = {
        l: {"h": torch.stack(layer_to_h_list[l], dim=0)}
        for l in sorted(layer_to_h_list.keys())
    }
    save_obj["y_softmax"] = y_softmax
    save_obj["model_output"] = model_outputs

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/"))
    out_path = os.path.join(args.output_dir, f"{model_name}-val_last_prompt_token.pt")
    torch.save(save_obj, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        default="val.json",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="minicpm-o-2_6",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="hiddenstates",
    )
    parser.add_argument("--max_new_tokens", type=int, default=8)
    args = parser.parse_args()
    run(args)

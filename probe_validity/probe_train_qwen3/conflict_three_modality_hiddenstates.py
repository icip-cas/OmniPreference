import json
import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info


def get_single_token_id(tokenizer, s: str) -> int:
    for cand in (s, " " + s, "\n" + s):
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]


def option_id_by_modality(sample):
    mod_to_opt = {}
    for opt in sample["options"]:
        mod_to_opt[opt["modality"]] = opt["option_id"]
    text_oid = mod_to_opt.get("text")
    image_oid = mod_to_opt.get("image")
    audio_oid = mod_to_opt.get("audio")
    return text_oid, image_oid, audio_oid


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
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)
    model.eval()

    tok = processor.tokenizer
    id_A = get_single_token_id(tok, "A")
    id_B = get_single_token_id(tok, "B")
    id_C = get_single_token_id(tok, "C")
    optid_to_tokid = {"A": id_A, "B": id_B, "C": id_C}
    print('optid_to_tokid:', optid_to_tokid)

    layer_to_h_list = None
    y_softmax = []
    model_outputs = []

    for sample in tqdm(data):
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
        inputs = inputs.to(model.device).to(model.dtype)

        prompt_len = inputs["input_ids"].shape[1]
        last_prompt_pos = prompt_len - 1

        with torch.no_grad():
            thinker_out = model.thinker(
                **inputs,
                use_audio_in_video=False,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = thinker_out.hidden_states
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
                device=thinker_out.logits.device,
            )

            next_token_logits = thinker_out.logits[0, -1, :].to(torch.float32)

            probs_full = F.softmax(next_token_logits, dim=-1)
            probs_tia = probs_full[tok_ids_in_mod_order].to("cpu")
            y_softmax.append(probs_tia)

            text_ids, _ = model.generate(
                **inputs,
                use_audio_in_video=False,
                speaker="Ethan",
                thinker_max_new_tokens=args.max_new_tokens,
                thinker_return_dict_in_generate=True,
            )

            gen_ids = text_ids.sequences[:, prompt_len:]
            response = processor.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            model_outputs.append(response)

    y_softmax = torch.stack(y_softmax, dim=0)

    save_obj = {l: {"h": torch.stack(layer_to_h_list[l], dim=0)} for l in sorted(layer_to_h_list.keys())}
    save_obj["y_softmax"] = y_softmax
    save_obj["model_output"] = model_outputs

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/"))
    out_path = os.path.join(args.output_dir, f"{model_name}-train_last_prompt_token.pt")
    torch.save(save_obj, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default='train.json')
    parser.add_argument("--model_path", type=str, default='Qwen3-Omni-30B-A3B-Instruct')
    parser.add_argument("--output_dir", type=str, default='hiddenstates')
    parser.add_argument("--max_new_tokens", type=int, default=8)
    args = parser.parse_args()
    run(args)

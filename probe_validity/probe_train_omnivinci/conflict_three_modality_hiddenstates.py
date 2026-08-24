import json
import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, AutoConfig


def get_single_token_id(tokenizer, s: str) -> int:
    for cand in (s, " " + s, "\n" + s):
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"Cannot encode '{s}' to a single token")


def option_id_by_modality(sample):
    mod_to_opt = {}
    for opt in sample["options"]:
        mod_to_opt[opt["modality"]] = opt["option_id"]
    return mod_to_opt.get("text"), mod_to_opt.get("image"), mod_to_opt.get("audio")


def build_messages(sample):
    text_label = sample["text"]
    image_path = sample["image"]
    audio_path = sample["audio"]
    options = sample["options"]

    options_dict = {opt["option_id"]: opt["label"] for opt in options}

    question = (
        "which option best describes what this example is mainly about?\n\n"
        f"A. {options_dict['A']}\n"
        f"B. {options_dict['B']}\n"
        f"C. {options_dict['C']}\n\n"
        "You should only output the single letter of your choice (A, B, or C), with no explanation or additional text."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_label},
                {"type": "image", "image": image_path},
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": question},
            ],
        },
    ]
    return messages


def run(args):
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    model.config.load_audio_in_video = False
    processor.config.load_audio_in_video = False
    model.eval()

    llm_backbone = None
    llm_attr_name = None
    for attr_name in ["llm", "language_model", "model"]:
        if hasattr(model, attr_name):
            candidate = getattr(model, attr_name)
            if hasattr(candidate, "forward") and hasattr(candidate, "config"):
                llm_backbone = candidate
                llm_attr_name = attr_name
                break

    assert llm_backbone is not None, (
        "Cannot find LLM backbone. Please inspect model structure:\n"
        f"  type: {type(model)}\n"
        f"  attrs: {[a for a in dir(model) if not a.startswith('_')]}\n"
        "Run print(model) to see the full architecture."
    )
    print(f"[INFO] LLM backbone: model.{llm_attr_name} ({type(llm_backbone).__name__})")

    assert hasattr(model, "_embed"), (
        "Cannot find model._embed(). "
        "This method is required to convert input_ids + media into inputs_embeds.\n"
        f"Available methods: {[a for a in dir(model) if not a.startswith('__')]}"
    )
    print("[INFO] model._embed() found — will use _embed + llm.forward (most reliable)")

    tok = processor.tokenizer
    id_A = get_single_token_id(tok, "A")
    id_B = get_single_token_id(tok, "B")
    id_C = get_single_token_id(tok, "C")
    optid_to_tokid = {"A": id_A, "B": id_B, "C": id_C}
    print("optid_to_tokid:", optid_to_tokid)

    layer_to_h_list = None
    y_softmax = []

    for idx, sample in enumerate(tqdm(data)):
        messages = build_messages(sample)

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor([text])

        input_ids = inputs.input_ids.to(model.device)
        media = getattr(inputs, "media", None)
        media_config = getattr(inputs, "media_config", None)

        with torch.no_grad():
            inputs_embeds, _, attention_mask = model._embed(
                input_ids, media, media_config,
                labels=None, attention_mask=None
            )

            out = llm_backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = out.hidden_states
            logits = out.logits

            last_prompt_pos = inputs_embeds.shape[1] - 1

            if idx == 0:
                print(f"[DEBUG] input_ids length (before expand): {input_ids.shape[1]}")
                print(f"[DEBUG] inputs_embeds length (after expand): {inputs_embeds.shape[1]}")
                print(f"[DEBUG] hidden_states seq_len: {hidden_states[-1].shape[1]}")
                print(f"[DEBUG] logits seq_len: {logits.shape[1]}")
                print(f"[DEBUG] last_prompt_pos: {last_prompt_pos}")
                print(f"[DEBUG] num_layers (incl emb): {len(hidden_states)}")
                print(f"[DEBUG] hidden_dim: {hidden_states[-1].shape[2]}")
                assert hidden_states[-1].shape[1] == inputs_embeds.shape[1] == logits.shape[1], (
                    f"Seq length mismatch! embeds={inputs_embeds.shape[1]}, "
                    f"hidden={hidden_states[-1].shape[1]}, logits={logits.shape[1]}"
                )

            num_layers_total = len(hidden_states)
            if layer_to_h_list is None:
                layer_to_h_list = {l: [] for l in range(1, num_layers_total)}

            for l in range(1, num_layers_total):
                h_last = hidden_states[l][:, last_prompt_pos, :]
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
                device=logits.device,
            )
            if idx == 0:
                print("tok_ids_in_mod_order:", tok_ids_in_mod_order)

            next_token_logits = logits[0, last_prompt_pos, :].to(torch.float32)
            probs_full = F.softmax(next_token_logits, dim=-1)
            probs_tia = probs_full[tok_ids_in_mod_order].to("cpu")
            y_softmax.append(probs_tia)

    y_softmax = torch.stack(y_softmax, dim=0)

    save_obj = {
        l: {"h": torch.stack(layer_to_h_list[l], dim=0)}
        for l in sorted(layer_to_h_list.keys())
    }
    save_obj["y_softmax"] = y_softmax

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/"))
    out_path = os.path.join(args.output_dir, f"{model_name}-val_last_prompt_token.pt")
    torch.save(save_obj, out_path)
    print(f"\nSaved: {out_path}")
    print(f"  Layers : {sorted(layer_to_h_list.keys())}")
    print(f"  Samples: {len(y_softmax)}")
    print(f"  Hidden dim: {layer_to_h_list[1][0].shape[-1]}")
    print(f"  y_softmax shape: {y_softmax.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OmniVinci: extract last-prompt-token hidden states + logits (text+image+audio)"
    )
    parser.add_argument("--data_file", type=str, default="val.json")
    parser.add_argument("--model_path", type=str, default="omnivinci")
    parser.add_argument("--output_dir", type=str, default="hiddenstates")
    args = parser.parse_args()
    run(args)

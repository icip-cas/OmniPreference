import json
import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor
from modeling_bailingmm import BailingMMNativeForConditionalGeneration


def get_single_token_id(tokenizer, s: str) -> int:
    for cand in (s, " " + s, "\n" + s):
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]


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
    optA, optB, optC = options_dict["A"], options_dict["B"], options_dict["C"]

    question = (
        "which option best describes what this example is mainly about?\n\n"
        f"A. {optA}\n"
        f"B. {optB}\n"
        f"C. {optC}\n\n"
        "You should only output the single letter of your choice (A, B, or C), "
        "with no explanation or additional text."
    )

    return [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": text_label},
                {"type": "image", "image": image_path},
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": question},
            ],
        },
    ]


def run(args):
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        load_image_gen=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()

    processor = AutoProcessor.from_pretrained(args.processor_path, trust_remote_code=True)

    tok = processor.tokenizer
    id_A = get_single_token_id(tok, "A")
    id_B = get_single_token_id(tok, "B")
    id_C = get_single_token_id(tok, "C")
    optid_to_tokid = {"A": id_A, "B": id_B, "C": id_C}
    print("optid_to_tokid:", optid_to_tokid)

    layer_to_h_list = None
    y_softmax = []

    for sample in tqdm(data):
        messages = build_messages(sample)

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
            audio_kwargs={"use_whisper_encoder": True},
        ).to(model.device)

        for k in inputs.keys():
            if k in ("pixel_values", "pixel_values_videos"):
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        prompt_len = input_ids.shape[1]
        last_prompt_pos = prompt_len - 1

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                image_embeds = None
                if inputs.get("pixel_values") is not None:
                    image_embeds = model.extract_image_feature(
                        inputs["pixel_values"], grid_thw=inputs.get("image_grid_thw")
                    )

                video_embeds = None
                if inputs.get("pixel_values_videos") is not None:
                    video_embeds = model.extract_image_feature(
                        inputs["pixel_values_videos"], grid_thw=inputs.get("video_grid_thw")
                    )

                audio_embeds, audio_embeds_lengths = None, None
                if inputs.get("audio_feats") is not None:
                    audio_embeds, audio_embeds_lengths = model.extract_audio_feature(
                        inputs["audio_feats"],
                        inputs.get("audio_feats_lengths"),
                        use_whisper_encoder=True,
                    )

                if (
                    image_embeds is None and video_embeds is None and audio_embeds is None
                ) or input_ids.size(1) == 1:
                    words_embeddings = model.model.get_input_embeddings()(
                        input_ids.clip(0, model.model.get_input_embeddings().weight.shape[0] - 1)
                    )
                    image_mask = None
                    audio_mask = None
                else:
                    words_embeddings, image_mask, audio_mask = model.prompt_wrap_navit(
                        input_ids.clip(0, model.model.get_input_embeddings().weight.shape[0] - 1),
                        image_embeds,
                        video_embeds,
                        audio_embeds,
                        audio_embeds_lengths,
                        inputs.get("audio_placeholder_loc_lens"),
                        None,
                    )

                position_ids = None
                if (
                    model.config.llm_config.rope_scaling is not None
                    and model.config.llm_config.rope_scaling.get("type") == "3D"
                ):
                    position_ids, _ = model.get_rope_index(
                        input_ids,
                        image_token_id=model.config.llm_config.image_patch_token,
                        video_token_id=model.config.llm_config.image_patch_token,
                        image_start_token_id=model.config.llm_config.image_start_token,
                        video_start_token_id=model.config.llm_config.video_start_token,
                        image_grid_thw=inputs.get("image_grid_thw"),
                        video_grid_thw=inputs.get("video_grid_thw"),
                        attention_mask=attention_mask,
                    )

                outputs = model.model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=words_embeddings,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                    image_mask=image_mask,
                    audio_mask=audio_mask,
                )

            hidden_states = outputs.hidden_states
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
                device=outputs.logits.device,
            )

            next_token_logits = outputs.logits[0, -1, :].to(torch.float32)
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
    out_path = os.path.join(
        args.output_dir, f"{model_name}-val_last_prompt_token.pt"
    )
    torch.save(save_obj, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", type=str,
        default="val.json",
    )
    parser.add_argument(
        "--model_path", type=str,
        default="Ming-Lite-Omni-1.5",
    )
    parser.add_argument(
        "--processor_path", type=str, default=".",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="hiddenstates",
    )
    args = parser.parse_args()
    run(args)

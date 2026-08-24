import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

MOD2IDX = {"text": 0, "vision": 1, "audio": 2}
DISTRACTOR_DEFAULT = ["audio"]
USE_AUDIO_IN_VIDEO = True
DS = "audio-driven"


def build_messages(sample, args):
    video_path = os.path.join(args.video_dir, f"{sample['video_id']}.mp4")
    prompt = (
        f"{sample['text']}\n"
        "A. Yes. B. No.\n"
        "Select the best option for the question."
    )
    return [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": prompt},
        ],
    }]


def get_decoder_layers(model):
    layers = model.thinker.model.layers
    assert len(layers) > 0, "model.thinker.model.layers is empty, please check the model.thinker structure"
    return layers


def load_probe_weight(probe_dir, peak_layer, device):
    sd = torch.load(os.path.join(probe_dir, f"layer_{peak_layer}.pt"), map_location="cpu")
    return sd["weight"].float().to(device)


def build_direction(W, distractor_idxs, contrastive=False):
    idxs = list(distractor_idxs)
    d = W[idxs].sum(dim=0)
    return d / (d.norm(p=2) + 1e-12)


def make_steering_hook(d_unit, coeff):
    def _steer(hs):
        dv = d_unit.to(device=hs.device, dtype=hs.dtype)
        h_last = hs[..., -1, :]
        steered = hs.clone()
        steered[..., -1, :] = h_last + coeff * dv
        return steered

    def hook(module, inputs, output):
        if coeff == 0.0:
            return output
        if isinstance(output, tuple):
            return (_steer(output[0]),) + tuple(output[1:])
        return _steer(output)
    return hook


def run(args):
    distractor = args.distractor or DISTRACTOR_DEFAULT
    distractor_idxs = [MOD2IDX[m] for m in distractor]
    distractor_tag = "+".join(distractor)
    coeff = float(args.coeff)

    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[{DS}] loaded {len(data)} samples | distractor={distractor_tag}(idx {distractor_idxs}) "
          f"| peak_layer={args.peak_layer} | coeff={coeff}")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model.eval()
    device = model.device

    W = load_probe_weight(args.probe_dir, args.peak_layer, device)
    d_unit = build_direction(W, distractor_idxs, contrastive=args.contrastive)

    layers = get_decoder_layers(model)
    hook_idx = args.peak_layer - 1
    assert 0 <= hook_idx < len(layers), f"peak_layer={args.peak_layer} is out of range (total {len(layers)} layers)"
    target_layer = layers[hook_idx]
    print(f"[{DS}] hook on layers[{hook_idx}] == hidden_states[{args.peak_layer}]")

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/"))

    hook = make_steering_hook(d_unit, coeff)
    handle = target_layer.register_forward_hook(hook)
    predictions = []
    try:
        for sample in tqdm(data, desc=f"{DS} coeff={coeff}"):
            messages = build_messages(sample, args)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(
                messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=False, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            prompt_len = inputs["input_ids"].shape[1]
            inputs = inputs.to(device, dtype=model.dtype)

            with torch.no_grad():
                gen = model.generate(
                    **inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO,
                    return_audio=False, max_new_tokens=args.max_new_tokens)
            response = processor.batch_decode(
                gen[:, prompt_len:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]

            rec = dict(sample)
            rec["model_raw_output"] = response
            predictions.append(rec)
    finally:
        handle.remove()

    coeff_tag = f"{coeff:g}"
    out_json = os.path.join(args.output_dir, f"{model_name}-{DS}-({coeff_tag}).json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)
    print(f"[{DS}] saved {len(predictions)} samples -> {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file",
                    default='avh-audio-driven-video-hallucination.json')
    ap.add_argument("--model_path",
                    default="Qwen2.5-Omni-7B")
    ap.add_argument("--probe_dir",
                    default='probe_softmax',
                    help="directory containing layer_{L}.pt probe weight files")
    ap.add_argument("--peak_layer", type=int, required=True, help="peak preference layer L (corresponds to hidden_states[L])")
    ap.add_argument("--distractor", choices=["text", "vision", "audio"], nargs="+", default=None,
                    help="distractor modality; default: audio")
    ap.add_argument("--coeff", type=float, default=0,
                    help="steering coefficient (single value); <0 suppresses, >0 enhances, 0 means no intervention (baseline)")
    ap.add_argument("--contrastive", action="store_true", help="use the W[distractor]-mean(others) direction")
    ap.add_argument("--video_dir", default="videos")
    ap.add_argument("--output_dir", default='results', help="directory to save the output json")
    ap.add_argument("--max_new_tokens", type=int, default=5)
    args = ap.parse_args()
    run(args)

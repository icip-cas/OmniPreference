<div align="center">

# Beyond Text-Dominance: Understanding Modality Preference of Omni-Modal Large Language Models
</div>

When text, vision, and audio conflict, most native omni-modal large language models prefer visual evidence. This preference emerges in the middle-to-late decoder layers, causally contributes to cross-modal hallucination, and can be reused as a zero-shot hallucination signal.

<p align="center">
  <img src="assets/overview.png" alt="Overview of Omni-Preference" width="100%">
</p>

## Overview

Native omni-modal large language models (OLLMs) map text, vision, and audio into a unified representation space, but they do not necessarily use these modalities equally. **Omni-Preference** studies what happens when the three modalities provide mutually conflicting semantic evidence.

This repository supports four connected experiments:

- **Preference evaluation:** construct a controlled trimodal conflict benchmark and quantify each model's modality preference with Modality Selection Rate (MSR).
- **Mechanistic probing:** train a linear probe at every decoder layer to locate when modality preference forms.
- **Hallucination analysis:** measure how abnormal preference for a distractor modality relates to cross-modal hallucination.
- **Causal intervention and detection:** steer probe-derived modality directions and reuse probe scores for zero-shot hallucination detection.

### Key Findings

- Eight of the ten evaluated OLLMs exhibit a clear visual preference, departing from the text dominance commonly reported for traditional vision-language models.
- Modality preference evolves through four stages - **Absence, Emergence, Peak, and Decline** - and is strongest in the middle-to-late decoder layers.
- Hallucinated samples have significantly higher probe-estimated preference for the task's distractor modality.
- Suppressing the distractor direction reduces hallucination, while amplifying it generally has the opposite effect.
- On CMM Audio Dominance, probe-based zero-shot hallucination detection reaches an average AUROC of **0.83** across Qwen2.5-Omni-7B, MiniCPM-o-2.6, and Qwen3-Omni.

## Contents

- [Main Results](#main-results)
- [Supported Models and Experiments](#supported-models-and-experiments)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Layer-wise Preference Probing](#layer-wise-preference-probing)
- [Hallucination Correlation and Detection](#hallucination-correlation-and-detection)
- [Causal Steering](#causal-steering)
- [Repository Structure](#repository-structure)
- [Reproducibility Notes](#reproducibility-notes)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Main Results

### Modality Preference Landscape

For modality $m$, the Modality Selection Rate is

$$
\mathrm{MSR}(m)=\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}\left[\hat{y}_i=\mathrm{opt}_i(m)\right].
$$

The uniform baseline for text, vision, and audio is $1/3$.

<p align="center">
  <img src="assets/tri_modal_msr.png" alt="Modality Selection Rate across ten OLLMs" width="78%">
</p>

| Model | Text | Vision | Audio |
| --- | ---: | ---: | ---: |
| Gemini 3.1 Pro | 7% | **72%** | 21% |
| Gemini 3 Flash | 17% | **70%** | 13% |
| Gemini 2.5 Pro | 36% | **50%** | 14% |
| Gemini 2.5 Flash | 28% | **68%** | 4% |
| OmniVinci | 27% | **72%** | 1% |
| MiniCPM-o-2.6 | 24% | **67%** | 9% |
| Qwen2.5-Omni-7B | 25% | **57%** | 18% |
| Qwen2.5-Omni-3B | 44% | **52%** | 4% |
| Ming-Lite-Omni 1.5 | **50%** | 48% | 2% |
| Qwen3-Omni-30B-A3B-Instruct | **54%** | 42% | 4% |

Values are rounded to the nearest percentage. The reported evaluation set contains 1,000 samples that every evaluated model answers correctly when each modality is presented independently.

### Layer-wise Preference Formation

For each decoder layer, we extract the last prompt-token hidden state, apply L2 normalization, and train a linear classifier with three outputs ordered as **text, vision, audio**. Each model uses a balanced pool of 3,000 examples with an 8:1:1 train/validation/test split.

<p align="center">
  <img src="assets/probe_train_pipeline.png" alt="Layer-wise preference probe training pipeline" width="80%">
</p>

<p align="center">
  <img src="assets/relative_depth_accuracy_curve.png" alt="Probe accuracy across relative decoder depth" width="72%">
</p>

Preference signals are weak in approximately the first 40% of layers, rise rapidly between 40% and 70% depth, peak around 70%-90%, and decline in the final layers. Qwen2.5-Omni-7B reaches a peak probe accuracy of 0.91.

### Preference and Cross-modal Hallucination

<p align="center">
  <img src="assets/qwen2.5_omni_7b_cmm_kde.png" alt="Distractor preference distributions on CMM" width="92%">
</p>

On Qwen2.5-Omni-7B, the Mann-Whitney U test confirms that hallucination and non-hallucination samples have significantly different distractor-preference distributions:

| CMM task | Distractor | p-value |
| --- | --- | ---: |
| Language Dominance | Text | $1.82\times10^{-6}$ |
| Visual Dominance | Vision | $3.99\times10^{-12}$ |
| Audio Dominance | Audio | $1.05\times10^{-30}$ |

The probe-estimated distractor probability also supports zero-shot hallucination detection:

| Task | Qwen2.5-Omni-7B | MiniCPM-o-2.6 | Qwen3-Omni |
| --- | ---: | ---: | ---: |
| CMM Language Dominance | 0.68 | 0.70 | 0.65 |
| CMM Visual Dominance | 0.73 | 0.62 | 0.57 |
| CMM Audio Dominance | **0.86** | **0.75** | **0.87** |
| AVHBench Video-driven Audio Hallucination | 0.57 | 0.57 | 0.52 |
| AVHBench Audio-driven Video Hallucination | 0.63 | 0.63 | 0.72 |

### Causal Steering

At the layer with the strongest preference signal, the probe-weight column for modality $m$ is treated as a steering direction $d_m$. The last-token hidden state is modified as

$$
h'=h+\alpha d_m.
$$

Negative $\alpha$ suppresses the selected direction and positive $\alpha$ amplifies it.

<p align="center">
  <img src="assets/qwen2.5_omni_7b_steering.png" alt="Qwen2.5-Omni-7B causal steering results" width="72%">
</p>

For Qwen2.5-Omni-7B, suppressing the visual distractor direction with $\alpha=-0.7$ improves AVHBench Video-driven Audio Hallucination accuracy from **72.14% to 81.53%**. Applying the same intervention to OmniBench improves accuracy from **49.47% to 56.14%**.

## Supported Models and Experiments

| Model | Preference | Layer-wise probe | CMM/AVH analysis | Steering |
| --- | :---: | :---: | :---: | :---: |
| Qwen2.5-Omni-3B | ✓ | ✓ | - | - |
| Qwen2.5-Omni-7B | ✓ | ✓ | ✓ | ✓ |
| Qwen3-Omni-30B-A3B-Instruct | ✓ | ✓ | ✓ | - |
| MiniCPM-o-2.6 | ✓ | ✓ | ✓ | - |
| OmniVinci | ✓ | ✓ | - | - |
| Ming-Lite-Omni 1.5 | ✓ | ✓ | - | - |
| Gemini 2.5 Pro / Flash | ✓ | - | - | - |
| Gemini 3.1 Pro / 3 Flash | ✓ | - | - | - |

The current released `casual_analysis/` directory contains the Qwen2.5-Omni-7B intervention implementation. The directory name is historical and intentionally preserved.

## Installation

We recommend Python 3.10, Linux, NVIDIA CUDA, and a recent PyTorch release. The paper's full-model experiments were run on an NVIDIA A100 80GB GPU; actual memory usage depends on the checkpoint, precision, attention implementation, and device mapping.

```bash
conda create -n omni-preference python=3.10 -y
conda activate omni-preference

# Install the PyTorch build matching your CUDA environment first.
pip install torch torchvision torchaudio

pip install transformers accelerate safetensors \
  qwen-omni-utils numpy scipy scikit-learn matplotlib tqdm \
  pillow librosa soundfile decord moviepy av openai
```

The Qwen probe-extraction and CMM scripts request FlashAttention 2 by default:

```bash
pip install flash-attn --no-build-isolation
```

MiniCPM-o, OmniVinci, Ming-Lite-Omni, and Qwen3-Omni may require additional dependencies or specific `transformers` revisions from their official repositories. Install the requirements of the model adapter you plan to run.

## Data Preparation

The repository contains code and paper figures. It does **not** include model checkpoints, benchmark media, generated conflict JSON files, hidden-state tensors, or trained probe checkpoints.

### Trimodal Conflict Data

The constructor expects an aligned XModBench-style JSON array. Each record must contain:

```json
{
  "id": "sample-id",
  "text": "canonical semantic label",
  "image": "/absolute/or/readable/path/to/image.jpg",
  "audio": "/absolute/or/readable/path/to/audio.wav"
}
```

It also requires:

- `evaluation_preference/category.txt`: the six semantic categories included in this repository.
- `text_label.json`: canonical labels and IDs.
- `text_label_processed.json`: natural-language text mapped by the same IDs.

The two label-mapping JSON files and the original XModBench media are external inputs and must be supplied locally.

### Hallucination Benchmarks

Download CMM, AVHBench, and OmniBench from their official sources. Ensure every media path in the task JSON files resolves in your environment. The distractor modality for each task is:

| Task | Distractor modality |
| --- | --- |
| CMM Language Dominance | Text |
| CMM Visual Dominance | Vision |
| CMM Audio Dominance | Audio |
| AVHBench Video-driven Audio Hallucination | Vision |
| AVHBench Audio-driven Video Hallucination | Audio |

## Repository Structure

```text
.
|-- assets/                  # Paper figures used in this README
|-- evaluation_preference/  # Conflict construction, model inference, MSR statistics
|-- probe_validity/         # Model-specific hidden-state extraction and layer probes
|-- correlation/            # CMM/AVHBench extraction, significance tests, AUROC
|-- casual_analysis/        # Probe-derived activation steering
`-- README.md
```

## Quick Start

### 1. Construct the Conflict Pool

```bash
python evaluation_preference/construct_tri_conflicrt_sample.py \
  --input /path/to/xmodbench_modality_path.json \
  --categories evaluation_preference/category.txt \
  --text-labels /path/to/text_label.json \
  --processed-text-labels /path/to/text_label_processed.json \
  --output outputs/modality_conflict_5000.json \
  --num-samples 5000 \
  --seed 20260801 \
  --include-question
```

The script balances all $\binom{6}{3}=20$ category triplets, modality assignments, and answer-option orders, then validates the generated set before writing it.

### 2. Build the Shared Unimodally Correct Set

Run each model on text-only, image-only, and audio-only versions of the conflict pool. Each result must retain `options` and add `model_raw_output`. Filter correct samples with:

```bash
python evaluation_preference/get_correct_sample.py \
  --input /path/to/text_only_results.json \
  --modality text \
  --output outputs/correct/model_text.json
```

Use `--modality image` and `--modality audio` for the other inputs, then intersect all retained files:

```bash
python evaluation_preference/get_common_correct_sample.py \
  --inputs \
    outputs/correct/model1_text.json \
    outputs/correct/model1_image.json \
    outputs/correct/model1_audio.json \
    outputs/correct/model2_text.json \
    outputs/correct/model2_image.json \
    outputs/correct/model2_audio.json \
  --output outputs/conflict_sample_1000.json
```

Continue the input list for every evaluated model. The paper reports results on the 1,000 samples retained by all ten models.

### 3. Evaluate Modality Preference

Example for Qwen2.5-Omni-7B:

```bash
python evaluation_preference/qwen-2.5-omni-7B_run_conflict_triplets.py \
  --data_file outputs/conflict_sample_1000.json \
  --model_path /path/to/Qwen2.5-Omni-7B \
  --output_dir outputs/preference/qwen2.5-7b \
  --max_new_tokens 8
```

Compute MSR:

```bash
python evaluation_preference/stat_modality_bias.py \
  --input outputs/preference/qwen2.5-7b/Qwen2.5-Omni-7B-weak-conflict-audio-results.json
```

The Qwen2.5-Omni-7B script preserves the historical `weak-conflict-audio-results` output suffix even when a regular conflict set is supplied.

For closed-source Gemini models, set an OpenRouter key without placing it in source files:

```bash
export OPENROUTER_API_KEY="your-key"

python evaluation_preference/gemini-2.5Pro_run_conflict_triplets.py \
  --data_file outputs/conflict_sample_1000.json \
  --model_name google/gemini-2.5-pro \
  --output_dir outputs/preference/gemini-2.5-pro
```

On Windows PowerShell, set the key with `$env:OPENROUTER_API_KEY="your-key"`.

## Layer-wise Preference Probing

Each directory under `probe_validity/` implements the same pipeline for a different model architecture:

1. Create a balanced 8:1:1 split.
2. Extract last prompt-token hidden states and option-token soft labels.
3. Train one three-way linear probe per decoder layer.
4. Evaluate layer-wise probe accuracy.

Example for Qwen2.5-Omni-7B:

```bash
cd probe_validity/probe_train_qwen2.5-7B
```

Set `INPUT_JSON` in `split_data.py` to a sufficiently large preference-result file, then run:

```bash
python split_data.py

python conflict_three_modality_hiddenstates.py \
  --data_file train.json \
  --model_path /path/to/Qwen2.5-Omni-7B \
  --output_dir hiddenstates/train

python conflict_three_modality_hiddenstates.py \
  --data_file val.json \
  --model_path /path/to/Qwen2.5-Omni-7B \
  --output_dir hiddenstates/val

python conflict_three_modality_hiddenstates.py \
  --data_file test.json \
  --model_path /path/to/Qwen2.5-Omni-7B \
  --output_dir hiddenstates/test
```

Train and evaluate all layer-wise probes:

```bash
python train_probe_layer.py \
  --train_pt hiddenstates/train/Qwen2.5-Omni-7B-test_last_prompt_token.pt \
  --val_pt hiddenstates/val/Qwen2.5-Omni-7B-test_last_prompt_token.pt \
  --output_dir probe_softmax \
  --epochs 200 \
  --batch_size 256 \
  --lr 1e-3

python test_probe_layer_acc.py \
  --test_pt hiddenstates/test/Qwen2.5-Omni-7B-test_last_prompt_token.pt \
  --probe_dir probe_softmax \
  --output_png test_acc_by_layer.png
```

The hidden-state extractor uses the fixed suffix `test_last_prompt_token.pt` for all three splits; separating the output directories prevents overwriting.

## Hallucination Correlation and Detection

Return to the repository root and extract hidden states for a downstream task:

```bash
cd ../../

python correlation/qwen2.5-omni-cmm-text-driven_response_hiddenstates.py \
  --data_file /path/to/cmm-language-driven.json \
  --model_path /path/to/Qwen2.5-Omni-7B \
  --output_dir outputs/correlation/cmm-language
```

Apply every available layer-wise probe:

```bash
python correlation/layer_all_pred.py \
  --input_pt outputs/correlation/cmm-language/Qwen2.5-Omni-7B-CMM-language-driven-hidden-states.pt \
  --results_json outputs/correlation/cmm-language/Qwen2.5-Omni-7B-CMM-language-driven-results.json \
  --probe_dir probe_validity/probe_train_qwen2.5-7B/probe_softmax \
  --output_json outputs/correlation/cmm-language/all_layers.json \
  --start_layer 1 \
  --end_layer 28
```

Test the distractor-preference distribution at the peak layer:

```bash
python correlation/compute_p_value.py \
  --input_json outputs/correlation/cmm-language/all_layers.json \
  --distractor_modality text \
  --layer PEAK_LAYER \
  --output_txt outputs/correlation/cmm-language/p_value.txt
```

Compute per-layer zero-shot hallucination AUROC:

```bash
python correlation/compute_AUROC_per_layer.py \
  --input_json outputs/correlation/cmm-language/all_layers.json \
  --probe_dir probe_validity/probe_train_qwen2.5-7B/probe_softmax \
  --distractor_modality text \
  --output_txt outputs/correlation/cmm-language/auroc_by_layer.txt
```

The AUROC script uses `visual` for the vision class, while the p-value and intervention scripts use `vision`. This historical command-line difference is preserved in the released code.

## Causal Steering

The intervention code is located under `casual_analysis/`. Example: suppress the visual distractor direction on AVHBench Video-driven Audio Hallucination.

```bash
python casual_analysis/qwen/AVH_Video_driven_intervention_response.py \
  --data_file /path/to/avh-video-driven-audio-hallucination.json \
  --video_dir /path/to/avhbench/videos \
  --model_path /path/to/Qwen2.5-Omni-7B \
  --probe_dir probe_validity/probe_train_qwen2.5-7B/probe_softmax \
  --peak_layer PEAK_LAYER \
  --distractor vision \
  --coeff -0.7 \
  --output_dir outputs/steering/qwen2.5-7b/video-driven
```

Sweep `--coeff` over `-0.7 -0.5 -0.3 0 0.3 0.5 0.7` to reproduce the intervention curve. The OmniBench scripts provide the corresponding general-capability control. Set the result path inside `casual_analysis/acc.py` or `casual_analysis/acc_omnibench.py` before running the accuracy aggregator.



## Citation

If you find this work useful, please consider citing:

```bibtex
@article{yan2026beyond,
  title={Beyond Text-Dominance: Understanding Modality Preference of Omni-modal Large Language Models},
  author={Yan, Xinru and Cao, Boxi and Lu, Yaojie and Lin, Hongyu and Zhou, Weixiang and Sun, Le and Han, Xianpei},
  journal={arXiv preprint arXiv:2604.16902},
  year={2026}
}
```

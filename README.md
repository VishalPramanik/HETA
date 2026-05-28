# HETA: Hessian-Enhanced Token Attribution for Interpreting Autoregressive LLMs

<p align="center">
  <a href="https://arxiv.org/abs/2604.13258"><img src="https://img.shields.io/badge/arXiv-2604.13258-b31b1b.svg" alt="arXiv"></a>
  <a href="https://iclr.cc/virtual/2026/poster/"><img src="https://img.shields.io/badge/ICLR-2026-blue.svg" alt="ICLR 2026"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
</p>

<p align="center">
  <strong>Official implementation</strong><br>
  <em>Hessian-Enhanced Token Attribution (HETA): Interpreting Autoregressive LLMs</em><br>
  <a href="https://arxiv.org/pdf/2604.13258">Vishal Pramanik</a><sup>1</sup>,
  <a href="#">Maisha Maliha</a><sup>2</sup>,
  <a href="#">Nathaniel D. Bastian</a><sup>3</sup>,
  <a href="#">Sumit Kumar Jha</a><sup>1</sup><br>
  <sup>1</sup>University of Florida &nbsp; <sup>2</sup>University of Oklahoma &nbsp; <sup>3</sup>United States Military Academy<br><br>
  Published as a conference paper at <strong>ICLR 2026</strong>
</p>

---

## Overview

Attribution methods seek to explain language model predictions by quantifying the contribution of input tokens to generated outputs. Most existing techniques rely on linear approximations that fail to capture the causal and semantic complexities of autoregressive generation. **HETA** addresses this with a principled three-component framework:

| Component | What It Captures | Paper Reference |
|---|---|---|
| **Semantic Transition Influence** | Causal attention–value flow from each input token to the target, enforcing directionality via the decoder's causal mask | Section 4, M_T |
| **Hessian-Based Sensitivity** | Second-order curvature of the log-likelihood surface, revealing nonlinear token interactions missed by gradient methods | Section 4, Eq. 2–3 |
| **KL Information Impact** | Distributional shift at the target position when a token is masked, providing a probabilistic measure of contribution | Section 4, Eq. 4 |

The final attribution score (Eq. 5):

```
Attr(x_i → x_T) = M_T[i] · ( β · S_i^(T) + γ · I(x_i → x_T) )
```

<p align="center">
  <em>Figure 1: Overview of the HETA pipeline. (a) Semantic transition influence via attention–value rollout,
  (b) Hessian-based curvature estimation via scalable HVPs, (c) KL-based information impact under token masking.
  See <a href="https://arxiv.org/pdf/2604.13258">the paper (Figure 1, Page 2)</a> for the full diagram.</em>
</p>

## Key Results

HETA consistently outperforms existing attribution methods across four decoder-only models (GPT-J 6B, Phi-3 14B, LLaMA-3.1 70B, Qwen2.5 3B) on three benchmark datasets and one curated evaluation set.

| Method | Soft-NC ↑ | Soft-NS ↑ | DSA ↑ |
|---|:---:|:---:|:---:|
| Integrated Gradients | 1.87 | 0.45 | −0.34 |
| Attention Rollout | 0.41 | −0.01 | −0.44 |
| ContextCite | 1.42 | 0.03 | −0.12 |
| ReAGent | 1.68 | 0.37 | 3.60 |
| **HETA (Ours)** | **10.3** | **2.31** | **4.80** |

*Results on GPT-J 6B / LongRA. See Tables 1 and 3 in the paper for full comparisons.*

---

## Installation

```bash
git clone https://github.com/VishalPramanik/HETA.git
cd HETA
pip install -e .
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, Transformers ≥ 4.36, CUDA-capable GPU recommended.

## Quick Start

### Python API

```python
from heta import HETA, HETAConfig

config = HETAConfig(beta=0.5, gamma=0.5, device="cuda")
attributor = HETA.from_pretrained("gpt2", config=config)

result = attributor.attribute(
    "Cam ordered a pizza and took it home. He had to use his chef knife to cut a slice"
)

# Top attributed context tokens → target "slice"
for token, score in result.topk(5):
    print(f"  {token:>12s}  {score:.4f}")
```

### Demo Script

```bash
python scripts/demo.py                           # GPT-2, CPU
python scripts/demo.py --model gpt2-medium --device cuda   # GPU
python scripts/demo.py --html attribution_viz.html         # HTML output
```

---

## Curated Attribution Dataset

We release a curated evaluation dataset (Section 5.1) for systematically evaluating token-level attribution quality. Each instance concatenates a **distractor** NarrativeQA-style segment with an **answer-bearing** SciQ-style support segment, followed by a question:

```
[NarrativeQA distractor] <s> [SciQ answer-bearing support] <s> Question: [SciQ question]
```

### Example

```
Input:
  "The protagonist returns to the village after the winter storm and
   reflects on her father's passing. <s> Photosynthesis primarily occurs
   in the leaves of the plant, where chloroplasts capture light energy.
   <s> Question: In which part of the plant does photosynthesis mainly
   take place?"

Target token:     "leaves"
Important words:  [leaves, photosynthesis, plant, chloroplasts, capture, light]
Distractor:       entire NarrativeQA segment (should receive LOW attribution)
```

A correct attribution method should concentrate mass on the SciQ support segment and suppress mass on the NarrativeQA distractor. The **DSA metric** quantifies this:

```
DSA = Σ (attribution on SciQ tokens) − Σ (attribution on NarrativeQA tokens)
```

### Dataset Files

| File | Records | Description |
|---|---|---|
| `data/heta_qa_dataset_100_unique.json` | 100 | All 10 × 10 unique narrative–SciQ combinations |
| `data/heta_qa_dataset_2000.json` | 2,000 | Expanded set for statistical evaluation |
| `data/build_qa_dataset.py` | — | Reproducible builder script |

See [`data/README.md`](data/README.md) for schema documentation and regeneration instructions.

### Running Evaluation on the Curated Dataset

```bash
# Quick test (first 5 examples, GPT-2)
python scripts/evaluate_curated.py --model gpt2 --max-examples 5

# Full evaluation on 100 unique examples
python scripts/evaluate_curated.py \
    --model gpt2 \
    --dataset data/heta_qa_dataset_100_unique.json \
    --output results/curated_gpt2.json

# Large-scale evaluation on GPU
python scripts/evaluate_curated.py \
    --model EleutherAI/gpt-j-6b \
    --dataset data/heta_qa_dataset_2000.json \
    --device cuda \
    --output results/curated_gptj.json
```

### Regenerating the Dataset

```bash
python data/build_qa_dataset.py --output data/heta_qa_dataset_100_unique.json
python data/build_qa_dataset.py --target-examples 2000 --output data/heta_qa_dataset_2000.json
```

---

## Evaluation on Benchmark Datasets

```bash
python scripts/run_attribution.py \
    --model gpt2 \
    --dataset tellmewhy \
    --beta 0.5 --gamma 0.5 \
    --device cuda \
    --output results/
```

Supported datasets: `longra`, `tellmewhy`, `wikibio`, `curated`.

---

## Project Structure

```
HETA/
├── heta/                            # Core library
│   ├── __init__.py                  # Package interface
│   ├── attribution.py               # Main HETA class (Eq. 5, Algorithm 1)
│   ├── semantic_flow.py             # Semantic transition influence (M_T)
│   ├── hessian_sensitivity.py       # Hessian-based sensitivity (S_i via HVPs)
│   ├── kl_divergence.py             # KL information impact (I_i)
│   ├── metrics.py                   # Soft-NC, Soft-NS, DSA, F1 alignment
│   ├── utils.py                     # Config, helpers, windowing
│   └── visualization.py             # Console + HTML visualisation
├── data/                            # Curated evaluation dataset
│   ├── README.md                    # Dataset documentation and schema
│   ├── build_qa_dataset.py          # Dataset builder script
│   ├── heta_qa_dataset_100_unique.json   # 100 unique examples
│   └── heta_qa_dataset_2000.json         # 2,000 examples (expanded)
├── scripts/
│   ├── demo.py                      # Quick demo with GPT-2
│   ├── run_attribution.py           # Benchmark dataset evaluation
│   └── evaluate_curated.py          # Curated dataset evaluation (DSA)
├── configs/
│   └── default.yaml                 # Default hyperparameters
├── tests/
│   └── test_heta.py                 # Unit + integration tests
├── requirements.txt
├── setup.py
├── LICENSE                          # Apache 2.0
└── README.md
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `beta` | 0.5 | Hessian sensitivity weight |
| `gamma` | 0.5 | KL information weight |
| `num_hvp_samples` | 10 | Hutchinson probe vectors (*m*) |
| `mask_scheme` | `"zero"` | Token masking: `"zero"` / `"mean"` / `"unk"` |
| `window_size` | 0 | Sliding window for long contexts (0 = off) |
| `window_overlap` | 0.5 | Fractional overlap between windows |
| `low_rank` | 0 | Low-rank Hessian rank (0 = full) |
| `layer_subset` | 0 | Curvature on last *n* layers (0 = all) |

### Model-Specific Recipes (from the paper)

| Model | Recommended Settings |
|---|---|
| GPT-2 / GPT-J 6B | Default (`low_rank=0`, all layers) |
| Phi-3 14B | `low_rank=64` |
| LLaMA-3.1 70B | `low_rank=64`, `layer_subset=6`, `window_size=512` |
| Qwen2.5 3B | Default |

## Supported Models

HETA works with any HuggingFace `AutoModelForCausalLM`-compatible decoder-only model:

| Model | Parameters | HuggingFace ID |
|---|---|---|
| GPT-2 | 124M | `gpt2` |
| GPT-J | 6B | `EleutherAI/gpt-j-6b` |
| Phi-3 Medium | 14B | `microsoft/Phi-3-medium-4k-instruct` |
| LLaMA 3.1 | 70B | `meta-llama/Llama-3.1-70B` |
| Qwen 2.5 | 3B | `Qwen/Qwen2.5-3B` |

## Tests

```bash
python -m pytest tests/ -v                                  # Unit tests
python -m pytest tests/test_heta.py::TestIntegration -v     # Integration (downloads GPT-2)
```

## Citation

```bibtex
@inproceedings{pramanik2026heta,
  title     = {Hessian-Enhanced Token Attribution ({HETA}): Interpreting Autoregressive {LLMs}},
  author    = {Pramanik, Vishal and Maliha, Maisha and Bastian, Nathaniel D. and Jha, Sumit Kumar},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2604.13258}
}
```

## Acknowledgments

This work was supported in part by the National Science Foundation under Grant No. 2404036, by University of Florida startup funds, and by DARPA under Contracts No. HR00112490420 and No. HR00112420004.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

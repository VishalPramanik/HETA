# Curated Attribution Dataset: NarrativeQA ⊕ SciQ

This directory contains the curated QA attribution evaluation dataset introduced
in **Section 5.1** of the HETA paper. The dataset is designed to test whether
attribution methods concentrate importance mass on truly predictive evidence
rather than semantically rich but non-diagnostic distractors.

## Construction

Each instance pairs two segments followed by a question:

```
[NarrativeQA-style distractor] <s> [SciQ answer-bearing support] <s> Question: [SciQ question]
```

The model generates the first answer token, and attributions are evaluated with
respect to this onset token. Correct attribution should place mass on the SciQ
support segment (which contains the answer) and suppress mass on the NarrativeQA
distractor (which is semantically rich but irrelevant to the question).

### Example

```
Input:
  "The protagonist returns to the village after the winter storm and
   reflects on her father's passing. <s> Photosynthesis primarily occurs
   in the leaves of the plant, where chloroplasts capture light energy.
   <s> Question: In which part of the plant does photosynthesis mainly
   take place?"

Target:       "leaves"
Important:    [leaves, photosynthesis, plant, chloroplasts, capture, light]
Distractor:   entire NarrativeQA segment
```

## Files

| File | Records | Description |
|---|---|---|
| `heta_qa_dataset_100_unique.json` | 100 | All 10 × 10 unique narrative–SciQ combinations |
| `heta_qa_dataset_2000.json` | 2,000 | Expanded set (cycles base combinations with shuffling) |
| `build_qa_dataset.py` | — | Reproducible builder script |

## Schema

Each example in the `"examples"` array contains:

| Field | Type | Description |
|---|---|---|
| `id` | str | Unique example identifier |
| `base_pair_id` | str | Underlying narrative–SciQ combination ID |
| `narrative_segment` | str | Distractor text (should receive low attribution) |
| `sciq_support_segment` | str | Answer-bearing evidence (should receive high attribution) |
| `question` | str | Question to be answered |
| `answer` | str | Gold answer string |
| `target_first_answer_token` | str | First token of the answer (attribution target) |
| `input_text` | str | Full model input (narrative + support + question) |
| `important_words` | list[str] | High-precision support words in the SciQ segment |
| `important_word_indices_in_sciq_support` | list[int] | Word-level positions of important words within the SciQ support |

## Regenerating

```bash
# All 100 unique combinations
python data/build_qa_dataset.py --output data/heta_qa_dataset_100_unique.json

# 2,000 examples (cycles base pairs)
python data/build_qa_dataset.py --target-examples 2000 --output data/heta_qa_dataset_2000.json
```

## Evaluation Metric: DSA

The **Dependent Sentence Attribution** (DSA) metric quantifies alignment:

```
DSA = Σ_{i ∈ S_SciQ} ss_i  −  Σ_{j ∈ S_NarrQA} fs_j
```

where attributions are normalised per-instance so total mass sums to 1.
Higher DSA indicates the method correctly concentrates mass on the
answer-bearing evidence. See `scripts/evaluate_curated.py` for usage.

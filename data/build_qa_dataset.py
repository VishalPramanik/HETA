#!/usr/bin/env python3
"""
Build a small HETA-style curated QA attribution dataset.

The paper's curated dataset pairs one irrelevant NarrativeQA-style narrative
segment with one answer-bearing SciQ-style support segment and then appends the
SciQ question. The important words are the answer and minimal support cues in the
second segment, not the distractor narrative segment.

This script uses small built-in example pools by default. You can replace the
NARRATIVE_ITEMS and SCIQ_ITEMS lists with real licensed NarrativeQA and SciQ
records, or load them from JSON files if you extend the loader.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "they", "this", "to", "was", "were", "where", "which", "who",
    "why", "what", "when", "how", "does", "do", "did", "mainly", "most",
    "with", "within", "through", "under", "over", "than", "then", "after",
    "before", "about", "also", "can", "will", "would", "could", "should",
}

NARRATIVE_ITEMS: List[str] = [
    "The protagonist returns to the village after the winter storm and reflects on her father's passing.",
    "A young traveler hides a letter inside his coat before crossing the crowded harbor at dusk.",
    "The detective studies the broken window while the household argues quietly in the hallway.",
    "After years away from home, the musician recognizes the old theater by the faded blue curtains.",
    "The captain orders the crew to lower the sails when the fog begins to cover the sea.",
    "A teacher finds an unfinished notebook on the classroom floor after the final bell rings.",
    "The family gathers around the kitchen table to discuss the strange map found in the attic.",
    "During the festival, a child follows the sound of drums through a narrow street filled with lanterns.",
    "The gardener waits beside the locked gate as rainwater gathers along the stone path.",
    "An old friend arrives unexpectedly and reminds the narrator of a promise made years earlier.",
]

SCIQ_ITEMS: List[Dict[str, str]] = [
    {
        "support": "Photosynthesis primarily occurs in the leaves of the plant, where chloroplasts capture light energy.",
        "question": "In which part of the plant does photosynthesis mainly take place?",
        "answer": "leaves",
    },
    {
        "support": "Evaporation changes liquid water into water vapor when heat provides enough energy for molecules to escape.",
        "question": "What process changes liquid water into water vapor?",
        "answer": "evaporation",
    },
    {
        "support": "The heart pumps blood through blood vessels, delivering oxygen and nutrients to tissues in the body.",
        "question": "Which organ pumps blood through the body?",
        "answer": "heart",
    },
    {
        "support": "Gravity is the force that pulls objects toward the center of Earth and gives objects weight.",
        "question": "What force pulls objects toward the center of Earth?",
        "answer": "gravity",
    },
    {
        "support": "A battery stores chemical energy and converts it into electrical energy when connected in a circuit.",
        "question": "What device stores chemical energy for use in a circuit?",
        "answer": "battery",
    },
    {
        "support": "The nucleus is the control center of a cell because it contains genetic material called DNA.",
        "question": "Which part of the cell contains DNA and acts as the control center?",
        "answer": "nucleus",
    },
    {
        "support": "Condensation forms clouds when water vapor cools and changes back into tiny liquid droplets.",
        "question": "What process forms clouds when water vapor cools?",
        "answer": "condensation",
    },
    {
        "support": "A magnet attracts iron because magnetic forces act on certain metals such as iron, nickel, and cobalt.",
        "question": "Which metal is attracted by a magnet in this example?",
        "answer": "iron",
    },
    {
        "support": "The lungs exchange oxygen and carbon dioxide during breathing, allowing the body to receive oxygen.",
        "question": "Which organs exchange oxygen and carbon dioxide during breathing?",
        "answer": "lungs",
    },
    {
        "support": "Friction is a force that opposes motion when two surfaces rub against each other.",
        "question": "What force opposes motion between rubbing surfaces?",
        "answer": "friction",
    },
]


def tokenize(text: str) -> List[str]:
    """Simple word tokenizer preserving alphanumeric terms."""
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text.lower())


def important_words(support: str, question: str, answer: str, max_words: int = 8) -> List[str]:
    """
    Extract high-precision support words from the answer-bearing SciQ segment.

    This approximates the paper's annotation idea without using external LLMs:
    1) always include answer tokens;
    2) include non-stopword overlaps between support and question;
    3) include nearby informative support words until max_words is reached.
    """
    support_tokens = tokenize(support)
    question_tokens = set(tokenize(question)) - STOPWORDS
    answer_tokens = tokenize(answer)

    selected: List[str] = []
    for tok in answer_tokens:
        if tok not in selected:
            selected.append(tok)

    for tok in support_tokens:
        if tok in question_tokens and tok not in selected and tok not in STOPWORDS:
            selected.append(tok)
        if len(selected) >= max_words:
            return selected

    for tok in support_tokens:
        if tok not in STOPWORDS and tok not in selected:
            selected.append(tok)
        if len(selected) >= max_words:
            break
    return selected


def word_indices(text: str, words: Sequence[str]) -> List[int]:
    """Return word-level positions in text whose normalized token is important."""
    wanted = set(w.lower() for w in words)
    return [i for i, tok in enumerate(tokenize(text)) if tok in wanted]


def first_answer_token(answer: str) -> str:
    toks = tokenize(answer)
    return toks[0] if toks else answer.strip().split()[0]


def build_examples(
    narratives: Sequence[str],
    sciq_items: Sequence[Dict[str, str]],
    target_examples: int | None = None,
    seed: int = 7,
) -> List[Dict[str, object]]:
    """
    Create HETA-style mixed-paragraph QA examples.

    If target_examples is None, all unique narrative × SciQ combinations are used.
    If target_examples is larger than the unique combination count, combinations are
    sampled with replacement to reach the requested size; the base_pair_id field
    records the underlying unique pair.
    """
    rng = random.Random(seed)
    pairs = [(ni, si) for ni in range(len(narratives)) for si in range(len(sciq_items))]
    if target_examples is None:
        chosen_pairs = pairs
    elif target_examples <= len(pairs):
        chosen_pairs = rng.sample(pairs, target_examples)
    else:
        chosen_pairs = [pairs[i % len(pairs)] for i in range(target_examples)]
        rng.shuffle(chosen_pairs)

    examples: List[Dict[str, object]] = []
    for ex_id, (ni, si) in enumerate(chosen_pairs):
        narrative = narratives[ni]
        item = sciq_items[si]
        support = item["support"]
        question = item["question"]
        answer = item["answer"]
        imp = important_words(support, question, answer)

        input_text = f"{narrative} <s> {support} <s> Question: {question}"
        examples.append(
            {
                "id": f"heta_qa_{ex_id:05d}",
                "base_pair_id": f"narrative_{ni:02d}__sciq_{si:02d}",
                "narrative_id": f"narrative_{ni:02d}",
                "sciq_id": f"sciq_{si:02d}",
                "narrative_segment": narrative,
                "sciq_support_segment": support,
                "question": question,
                "answer": answer,
                "target_first_answer_token": first_answer_token(answer),
                "input_text": input_text,
                "important_words": imp,
                "important_word_indices_in_sciq_support": word_indices(support, imp),
                "label_note": "Important words are restricted to the answer-bearing SciQ support segment; the NarrativeQA-style segment is a distractor.",
            }
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a HETA-style NarrativeQA+SciQ QA attribution dataset.")
    parser.add_argument("--output", type=Path, default=Path("heta_qa_dataset.json"), help="Output JSON path.")
    parser.add_argument("--target-examples", type=int, default=None, help="Number of examples to create. Omit for all unique combinations.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used for sampling/shuffling.")
    args = parser.parse_args()

    examples = build_examples(NARRATIVE_ITEMS[:10], SCIQ_ITEMS[:10], args.target_examples, args.seed)
    payload = {
        "dataset_name": "heta_style_narrativeqa_sciq_qa_attribution",
        "description": "Toy HETA-style dataset pairing a distractor narrative sentence with an answer-bearing SciQ support sentence and question.",
        "num_examples": len(examples),
        "num_unique_base_pairs": len({ex["base_pair_id"] for ex in examples}),
        "construction": "[NarrativeQA-style sentence] <s> [SciQ-style support sentence] <s> Question: [SciQ question]",
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()

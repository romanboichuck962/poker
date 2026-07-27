"""Evaluate a trained artifact on selected benchmark release dates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker44.validator.payload_view import prepare_hand_for_miner
from super_poker.dataset import load_examples
from super_poker.inference import SuperPokerModel
from super_poker.scoring import metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../Poker44-subnet/data/raw"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/super_poker_3.joblib"))
    parser.add_argument("--dates", help="Comma-separated dates; defaults to all releases")
    args = parser.parse_args()
    selected = {value.strip() for value in (args.dates or "").split(",") if value.strip()}
    examples = [e for e in load_examples(args.data_dir) if not selected or e.source_date in selected]
    model = SuperPokerModel(args.artifact)
    # Benchmark files contain fields and precision that miners never receive.
    # Project through the exact validator payload before standalone evaluation.
    chunks = [
        [prepare_hand_for_miner(hand) for hand in example.hands]
        for example in examples
    ]
    scores = model.predict_chunk_scores(chunks)
    result = metrics([example.label for example in examples], scores)
    result["example_count"] = len(examples)
    result["dates"] = sorted({example.source_date for example in examples})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

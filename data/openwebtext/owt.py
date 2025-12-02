"""Utility to materialize OpenWebText into train/val txt files."""

import argparse
from pathlib import Path
from typing import Callable, Optional

from datasets import load_dataset
from tqdm.auto import tqdm

DEFAULT_VAL_RATIO = 5e-4
DEFAULT_SEED = 2357


def _text_from_example(example: dict) -> str:
    """Return cleaned text for a dataset example."""
    return (example.get("text") or "").strip()


def _write_split(
    dataset_split,
    output_path: Path,
    *,
    text_extractor: Callable[[dict], str],
    max_samples: Optional[int] = None,
) -> None:
    """Write the provided dataset split into a newline-separated txt file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(dataset_split) if hasattr(dataset_split, "__len__") else None
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, example in enumerate(tqdm(dataset_split, desc=f"Writing {output_path.name}", total=total)):
            if max_samples is not None and idx >= max_samples:
                break
            text = text_extractor(example)
            if not text:
                continue
            handle.write(text + "\n")


def prepare_openwebtext(
    output_dir: Path,
    *,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> None:
    """Download, split, and serialize OpenWebText."""
    dataset = load_dataset("openwebtext", split="train")
    split_dataset = dataset.train_test_split(test_size=val_ratio, seed=seed, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")

    _write_split(
        split_dataset["train"],
        output_dir / "train.txt",
        text_extractor=_text_from_example,
        max_samples=max_train_samples,
    )
    _write_split(
        split_dataset["val"],
        output_dir / "val.txt",
        text_extractor=_text_from_example,
        max_samples=max_val_samples,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OpenWebText for minGemma")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where train.txt/val.txt will be written.",
    )
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for splitting.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on the number of training samples (for smoke tests).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional cap on the number of validation samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_openwebtext(
        args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )


if __name__ == "__main__":
    main()
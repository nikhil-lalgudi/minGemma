"""IMDb dataset preparation script."""

import argparse
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm.auto import tqdm


def _text_from_example(example: dict) -> str:
    return (example.get("text") or "").strip()


def _write_split(dataset_split, output_path: Path, *, max_samples: Optional[int] = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(dataset_split) if hasattr(dataset_split, "__len__") else None
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, example in enumerate(tqdm(dataset_split, desc=f"Writing {output_path.name}", total=total)):
            if max_samples is not None and idx >= max_samples:
                break
            text = _text_from_example(example)
            if not text:
                continue
            handle.write(text + "\n")


def prepare_imdb(
    output_dir: Path,
    *,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> None:
    dataset = load_dataset("stanfordnlp/imdb")
    train_split = dataset["train"]
    val_split = dataset.get("test") or dataset.get("validation")
    if val_split is None:
        raise ValueError("IMDb dataset is missing the test/validation split.")

    _write_split(train_split, output_dir / "train.txt", max_samples=max_train_samples)
    _write_split(val_split, output_dir / "val.txt", max_samples=max_val_samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare IMDb reviews for minGemma")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to store train.txt and val.txt",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap on training samples")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional cap on validation samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_imdb(
        args.output_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )


if __name__ == "__main__":
    main()
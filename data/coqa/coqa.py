"""Prepare the CoQA dataset for minGemma."""

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import load_dataset
from tqdm.auto import tqdm


def _clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        # CoQA uses "input_text" for both Qs and As
        return (value.get("input_text") or value.get("text") or "").strip()
    return str(value).strip()


def _answers_list(raw_answers) -> List[str]:
    if raw_answers is None:
        return []
    if isinstance(raw_answers, dict):
        if "input_text" in raw_answers:
            return [
                _clean_text(answer)
                for answer in raw_answers["input_text"]
            ]
        if "text" in raw_answers:
            return [_clean_text(answer) for answer in raw_answers["text"]]
    if isinstance(raw_answers, Iterable) and not isinstance(raw_answers, (str, bytes)):
        return [_clean_text(answer) for answer in raw_answers]
    return [_clean_text(raw_answers)]


def _format_example(example: dict) -> str:
    story = _clean_text(example.get("story"))
    questions = example.get("questions") or []
    answers = _answers_list(example.get("answers"))
    qa_lines: List[str] = []
    for idx, question in enumerate(questions):
        q_text = _clean_text(question)
        a_text = answers[idx] if idx < len(answers) else ""
        if not q_text and not a_text:
            continue
        qa_lines.append(f"Q: {q_text}\nA: {a_text}")
    sections = [section for section in [story, "\n".join(qa_lines)] if section]
    return "\n\n".join(sections)


def _write_split(dataset_split, output_path: Path, *, max_samples: Optional[int] = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(dataset_split) if hasattr(dataset_split, "__len__") else None
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, example in enumerate(tqdm(dataset_split, desc=f"Writing {output_path.name}", total=total)):
            if max_samples is not None and idx >= max_samples:
                break
            formatted = _format_example(example)
            if not formatted:
                continue
            handle.write(formatted + "\n")


def prepare_coqa(
    output_dir: Path,
    *,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> None:
    dataset = load_dataset("stanfordnlp/coqa")
    train_split = dataset["train"]
    val_split = dataset.get("validation") or dataset.get("test")
    if val_split is None:
        raise ValueError("CoQA dataset lacks a validation split.")

    _write_split(train_split, output_dir / "train.txt", max_samples=max_train_samples)
    _write_split(val_split, output_dir / "val.txt", max_samples=max_val_samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CoQA data files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to write train.txt and val.txt",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap on training samples")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional cap on validation samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_coqa(
        args.output_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )


if __name__ == "__main__":
    main()
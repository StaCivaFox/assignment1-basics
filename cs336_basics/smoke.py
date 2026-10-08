"""Connect the student's tokenizer and training loop for a tiny TinyStories run.

Run from the project directory: uv run python -m cs336_basics.smoke
Only complete stories from the beginning of the source file are read.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shlex
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from cs336_basics import train, train_bpe
from cs336_basics.tokenizer import Tokenizer

SPECIAL_TOKEN = "<|endoftext|>"
CONTEXT_LENGTH = 32


def sample_stories(source: Path, train_count: int, val_count: int) -> tuple[str, str]:
    """Split a small prefix into disjoint sets of complete, delimited stories."""
    stories: list[str] = []
    lines: list[str] = []
    needed = train_count + val_count
    with source.open(encoding="utf-8", newline="") as stream:
        for line in stream:
            lines.append(line)
            if line.strip() == SPECIAL_TOKEN:
                stories.append("".join(lines))
                lines = []
                if len(stories) == needed:
                    break
    if len(stories) < needed:
        raise ValueError(f"Requested {needed} complete stories, but found only {len(stories)} in {source}")
    return "".join(stories[:train_count]), "".join(stories[train_count:])


def save_tokens(tokenizer: Tokenizer, text: str, path: Path) -> int:
    """Check the student's encoding and save IDs in the training CLI's format."""
    ids = tokenizer.encode(text)
    if tokenizer.decode(ids) != text:
        raise ValueError(f"Tokenizer round-trip failed for {path.name}")
    if len(ids) <= CONTEXT_LENGTH:
        raise ValueError(f"{path.name} needs more than {CONTEXT_LENGTH} tokens")
    if min(ids) < 0 or max(ids) >= len(tokenizer.vocab):
        raise ValueError(f"Token IDs outside the vocabulary in {path.name}")
    np.save(path, np.asarray(ids, dtype=np.uint16), allow_pickle=False)
    return len(ids)


def main(argv: list[str] | None = None) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", type=Path, default=project_dir / "data" / "TinyStories-train-100MB.txt")
    parser.add_argument(
        "--work-dir", type=Path, help="New directory for samples, tokenizer, arrays, and training outputs"
    )
    parser.add_argument("--train-stories", type=int, default=100)
    parser.add_argument("--val-stories", type=int, default=20)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--prepare-only", action="store_true", help="Create artifacts and print the training command")
    args = parser.parse_args(argv)
    if not args.source.is_file():
        parser.error(f"Source file does not exist: {args.source}")
    if args.train_stories < 1 or args.val_stories < 1:
        parser.error("Both story counts must be positive")
    if not 257 <= args.vocab_size <= 65536:
        parser.error("Require 257 <= vocab-size <= 65536 for this uint16 smoke test")
    if args.steps < 2:
        parser.error("Use at least two steps with the one-step warmup")
    work_dir = args.work_dir or project_dir / "runs" / f"smoke-{datetime.now():%Y%m%d-%H%M%S-%f}"
    work_dir = work_dir.resolve()
    if work_dir.exists():
        parser.error(f"Choose a new --work-dir; this directory already exists: {work_dir}")

    torch.set_num_threads(2)
    print(f"Outputs: {work_dir}", flush=True)
    train_text, val_text = sample_stories(args.source, args.train_stories, args.val_stories)
    work_dir.mkdir(parents=True)
    train_text_path = work_dir / "train.txt"
    train_text_path.write_text(train_text, encoding="utf-8", newline="")
    (work_dir / "val.txt").write_text(val_text, encoding="utf-8", newline="")
    print(f"Selected {args.train_stories} training stories and {args.val_stories} validation stories.", flush=True)

    print(f"Training your BPE tokenizer with {args.vocab_size} vocabulary entries...", flush=True)
    vocab, merges = train_bpe.train_bpe(str(train_text_path), args.vocab_size, [SPECIAL_TOKEN])
    vocab_path = work_dir / "vocab.pkl"
    merges_path = work_dir / "merges.pkl"
    with vocab_path.open("wb") as stream:
        pickle.dump(vocab, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with merges_path.open("wb") as stream:
        pickle.dump(merges, stream, protocol=pickle.HIGHEST_PROTOCOL)
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, [SPECIAL_TOKEN])
    if set(tokenizer.vocab) != set(range(len(tokenizer.vocab))):
        raise ValueError("The training CLI requires contiguous vocabulary IDs starting at zero")

    train_array_path = work_dir / "train.npy"
    val_array_path = work_dir / "val.npy"
    train_token_count = save_tokens(tokenizer, train_text, train_array_path)
    val_token_count = save_tokens(tokenizer, val_text, val_array_path)
    print(f"Round-trip checks passed. Tokens: train={train_token_count}, validation={val_token_count}.", flush=True)

    training_args = [
        "--train-data",
        str(train_array_path),
        "--val-data",
        str(val_array_path),
        "--run-dir",
        str(work_dir / "training"),
        "--vocab-size",
        str(len(tokenizer.vocab)),
        "--device",
        "cpu",
        "--d-model",
        "32",
        "--num-layers",
        "1",
        "--num-heads",
        "4",
        "--d-ff",
        "64",
        "--context-length",
        str(CONTEXT_LENGTH),
        "--batch-size",
        "2",
        "--steps",
        str(args.steps),
        "--warmup-steps",
        "1",
        "--log-every",
        "1",
        "--eval-every",
        "2",
        "--eval-batches",
        "2",
        "--checkpoint-every",
        "3",
    ]
    manifest = {
        "source": str(args.source.resolve()),
        "train_stories": args.train_stories,
        "val_stories": args.val_stories,
        "special_tokens": [SPECIAL_TOKEN],
        "vocab_size": len(tokenizer.vocab),
        "merges": len(merges),
        "token_dtype": "uint16",
        "train_tokens": train_token_count,
        "val_tokens": val_token_count,
        "training_args": training_args,
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        "Training command:\n" + shlex.join(["uv", "run", "python", "-m", "cs336_basics.train", *training_args]),
        flush=True,
    )
    if args.prepare_only:
        return

    print("Running your training loop...", flush=True)
    train.main(training_args)
    checkpoint_path = work_dir / "training" / f"checkpoint{args.steps}"
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Training returned without the final checkpoint: {checkpoint_path}")
    print(f"Smoke run complete. Final checkpoint: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()

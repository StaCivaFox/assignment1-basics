"""File-based stages reused by the Modal quickstart and local smoke tests."""

from __future__ import annotations

import io
import json
import pickle
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO
from urllib.request import urlopen

SPECIAL_TOKEN = "<|endoftext|>"
SOURCES = {
    split: f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{split}.txt"
    for split in ("train", "valid")
}


def write_json(path: Path, values: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_settings(path: Path, settings: dict) -> None:
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError(f"{path}: settings differ; use a new dataset name for a different preparation")


def iter_documents(stream: TextIO, chunk_chars: int = 1 << 20) -> Iterator[str]:
    """Keep delimiters and whitespace, including delimiters split across read boundaries."""
    pending = ""
    while chunk := stream.read(chunk_chars):
        parts = (pending + chunk).split(SPECIAL_TOKEN)
        for part in parts[:-1]:
            yield part + SPECIAL_TOKEN
        pending = parts[-1]
    if pending:
        yield pending


def copy_stories(stream: TextIO, destination: Path, count: int) -> None:
    """Copy exactly count complete stories, retaining the EOS delimiters."""
    copied = 0
    with destination.open("w", encoding="utf-8", newline="") as output:
        for document in iter_documents(stream):
            if not document.endswith(SPECIAL_TOKEN):
                continue
            output.write(document)
            copied += 1
            if copied == count:
                break
    if copied != count:
        raise ValueError(f"Requested {count} complete stories, found {copied}")


def download_dataset(dataset_dir: Path, train_stories: int, val_stories: int) -> dict:
    """Use separate official train/validation sources; zero means download the full split."""
    if train_stories < 0 or val_stories < 0:
        raise ValueError("Story limits must be nonnegative")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    settings = {"sources": SOURCES, "train_stories": train_stories, "val_stories": val_stories}
    manifest_path = dataset_dir / "sources.json"
    check_settings(manifest_path, settings)
    write_json(manifest_path, settings)
    for split, count in (("train", train_stories), ("valid", val_stories)):
        path = dataset_dir / f"{split}.txt"
        if path.exists():
            print(f"Reusing {path}", flush=True)
            continue
        print(f"Downloading {split}: {'all' if count == 0 else count} stories", flush=True)
        temporary = path.with_suffix(".txt.partial")
        with urlopen(SOURCES[split], timeout=120) as response:
            if count:
                with io.TextIOWrapper(response, encoding="utf-8", newline="") as stream:
                    copy_stories(stream, temporary, count)
            else:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1 << 20)
        temporary.replace(path)
    return settings


def fit_bpe(dataset_dir: Path, vocab_size: int = 10000, bpe_stories: int = 0) -> dict:
    """Call the student's BPE implementation on training text only."""
    from cs336_basics.train_bpe import train_bpe

    if not 257 <= vocab_size <= 65536 or bpe_stories < 0:
        raise ValueError("Require 257 <= vocab_size <= 65536 and bpe_stories >= 0")
    settings = {"vocab_size": vocab_size, "bpe_stories": bpe_stories, "special_tokens": [SPECIAL_TOKEN]}
    manifest_path = dataset_dir / "tokenizer.json"
    check_settings(manifest_path, settings)
    if manifest_path.exists() and all((dataset_dir / name).exists() for name in ("vocab.pkl", "merges.pkl")):
        return settings
    source = dataset_dir / "train.txt"
    if bpe_stories:
        sample = dataset_dir / "bpe_train.txt"
        with source.open(encoding="utf-8", newline="") as stream:
            copy_stories(stream, sample, bpe_stories)
        source = sample
    print(f"Fitting a {vocab_size}-token BPE on {source}", flush=True)
    vocab, merges = train_bpe(str(source), vocab_size, [SPECIAL_TOKEN])
    for name, value in (("vocab.pkl", vocab), ("merges.pkl", merges)):
        path = dataset_dir / name
        temporary = path.with_suffix(".pkl.partial")
        with temporary.open("wb") as stream:
            pickle.dump(value, stream)
        temporary.replace(path)
    write_json(manifest_path, settings)
    return settings


def load_tokenizer(dataset_dir: Path):
    from cs336_basics.tokenizer import Tokenizer

    return Tokenizer.from_files(dataset_dir / "vocab.pkl", dataset_dir / "merges.pkl", [SPECIAL_TOKEN])


def encode_file(tokenizer, source: Path, destination: Path, batch_chars: int = 1 << 20) -> int:
    """Stream uint16 IDs to disk, batching at EOS boundaries to preserve BPE segmentation."""
    import numpy as np

    token_count = 0
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with source.open(encoding="utf-8", newline="") as stream, temporary.open("wb") as output:
        documents = []
        char_count = 0

        def flush_batch():
            text = "".join(documents)
            ids = tokenizer.encode(text)
            if token_count == 0 and tokenizer.decode(ids) != text:
                raise ValueError(f"Tokenizer round-trip failed for {source}")
            if ids and not (0 <= min(ids) <= max(ids) < len(tokenizer.vocab) <= 65536):
                raise ValueError("Token IDs must fit in uint16 and belong to the vocabulary")
            np.asarray(ids, dtype=np.uint16).tofile(output)
            return len(ids)

        for document in iter_documents(stream):
            documents.append(document)
            char_count += len(document)
            if char_count >= batch_chars:
                token_count += flush_batch()
                documents.clear()
                char_count = 0
        if documents:
            token_count += flush_batch()
    temporary.replace(destination)
    return token_count


def tokenize_dataset(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "tokens.json"
    if manifest_path.exists() and all((dataset_dir / f"{split}.bin").exists() for split in ("train", "valid")):
        return json.loads(manifest_path.read_text())
    tokenizer = load_tokenizer(dataset_dir)
    counts = {}
    for split in ("train", "valid"):
        print(f"Tokenizing {split} with the training BPE", flush=True)
        counts[split] = encode_file(tokenizer, dataset_dir / f"{split}.txt", dataset_dir / f"{split}.bin")
        print(f"{split}: {counts[split]:,} tokens", flush=True)
    manifest = {"dtype": "uint16", "vocab_size": len(tokenizer.vocab), "token_counts": counts}
    write_json(manifest_path, manifest)
    return manifest


def training_arguments(dataset_dir: Path, run_dir: Path, extra_args: list[str], device: str = "cuda") -> list[str]:
    """Use the Section 7 model dimensions; allow the existing training CLI's hyperparameter flags."""
    # These paths and dimensions must match this prepared dataset and output directory.
    reserved = {"--train-data", "--val-data", "--token-dtype", "--vocab-size", "--run-dir", "--device"}
    if any(arg.split("=", 1)[0] in reserved for arg in extra_args):
        raise ValueError(f"Pipeline sets these flags itself: {', '.join(sorted(reserved))}")
    metadata = json.loads((dataset_dir / "tokens.json").read_text())
    return [
        "--train-data", str(dataset_dir / "train.bin"),
        "--val-data", str(dataset_dir / "valid.bin"),
        "--token-dtype", metadata["dtype"], "--vocab-size", str(metadata["vocab_size"]),
        "--run-dir", str(run_dir), "--device", device,
        "--d-model", "512", "--num-layers", "4", "--num-heads", "16", "--d-ff", "1344",
        "--context-length", "256", "--rope-theta", "10000",
        "--batch-size", "32", "--steps", "5000", "--warmup-steps", "100",
        "--log-every", "10", "--eval-every", "100", "--eval-batches", "10",
        "--checkpoint-every", "500", "--experiment-name", run_dir.name,
        *extra_args,
    ]


def generate_text(
    run_dir: Path, prompt: str, max_new_tokens: int = 256, temperature: float = 0.8,
    top_p: float = 0.9, seed: int = 42, checkpoint_step: int = 0, device: str = "cuda",
) -> dict:
    import torch

    from cs336_basics.model import TransformerLM

    config = json.loads((run_dir / "config.json").read_text())
    dataset_dir = Path(json.loads((run_dir / "pipeline.json").read_text())["dataset_dir"])
    tokenizer = load_tokenizer(dataset_dir)
    if checkpoint_step:
        checkpoint_path = run_dir / f"checkpoint{checkpoint_step}"
    else:
        checkpoints = [p for p in run_dir.glob("checkpoint*") if p.name.removeprefix("checkpoint").isdigit()]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints in {run_dir}")
        checkpoint_path = max(checkpoints, key=lambda p: int(p.name.removeprefix("checkpoint")))
    keys = ("d_model", "num_heads", "d_ff", "vocab_size", "context_length", "num_layers", "rope_theta")
    model = TransformerLM(**{key: config[key] for key in keys})
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    eos_id = next(token_id for token_id, token in tokenizer.vocab.items() if token == SPECIAL_TOKEN.encode())
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Provide a nonempty prompt")
    torch.manual_seed(seed)
    output = model.generate(
        torch.tensor(prompt_ids, dtype=torch.long, device=device), max_new_tokens,
        temperature=temperature, top_p=top_p, eos_token_id=eos_id,
    )[0].tolist()
    result = {
        "checkpoint": checkpoint_path.name, "prompt": prompt, "temperature": temperature,
        "top_p": top_p, "seed": seed, "max_new_tokens": max_new_tokens,
        "generated_token_ids": output[len(prompt_ids):],
        "generated_token_count": len(output) - len(prompt_ids),
        "text": tokenizer.decode(output),
    }
    stem = f"generation-{datetime.now(UTC):%Y%m%d-%H%M%S-%f}"
    write_json(run_dir / f"{stem}.json", result)
    (run_dir / f"{stem}.txt").write_text(result["text"] + "\n", encoding="utf-8")
    print(result["text"], flush=True)
    print(f"Generated {result['generated_token_count']} tokens; saved {stem}.txt", flush=True)
    return result

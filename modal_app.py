"""Modal entrypoints: download -> bpe -> tokenize -> train -> generate.

Run from the repository root. See README.md for smoke and Section 7 commands.
"""

import logging
import os
import re
import shlex
import shutil
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
VOLUME_NAME = "cs336-section7"
DATA = Path("/data")
GPU = os.environ.get("CS336_MODAL_GPU", "L40S")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT))
    .uv_pip_install("matplotlib>=3.10,<4")
    .env({"CS336_MODAL_GPU": GPU, "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("cs336_basics")
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("cs336-section7", image=image, volumes={str(DATA): volume})


def named_path(category: str, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Use only letters, numbers, underscores, and hyphens in dataset/run names")
    return DATA / category / name


@app.function(cpu=1, memory=2048, timeout=4 * 60 * 60)
def download(dataset: str = "smoke", train_stories: int = 1000, val_stories: int = 100):
    from cs336_basics.pipeline import download_dataset

    volume.reload()
    try:
        return download_dataset(named_path("datasets", dataset), train_stories, val_stories)
    finally:
        volume.commit()


@app.function(cpu=2, memory=32768, timeout=24 * 60 * 60)
def bpe(dataset: str = "smoke", vocab_size: int = 512, bpe_stories: int = 0):
    from cs336_basics.pipeline import fit_bpe

    volume.reload()
    result = fit_bpe(named_path("datasets", dataset), vocab_size, bpe_stories)
    volume.commit()
    print(result)
    return result


@app.function(cpu=2, memory=8192, timeout=24 * 60 * 60)
def tokenize(dataset: str = "smoke"):
    from cs336_basics.pipeline import tokenize_dataset

    volume.reload()
    result = tokenize_dataset(named_path("datasets", dataset))
    volume.commit()
    return result


@app.function(gpu=GPU, cpu=4, memory=16384, timeout=4 * 60 * 60, scaledown_window=2)
def train(dataset: str, run_name: str, train_args: str = ""):
    from cs336_basics import pipeline, train as training

    volume.reload()
    dataset_dir = named_path("datasets", dataset)
    run_dir = named_path("runs", run_name)
    if run_dir.exists():
        raise ValueError(f"{run_dir} already exists; use a new run name, including when resuming")
    argv = pipeline.training_arguments(dataset_dir, run_dir, shlex.split(train_args))
    training.parse_args(argv)  # Check data and arguments before creating the run directory.
    run_dir.mkdir(parents=True)
    pipeline.write_json(run_dir / "pipeline.json", {"dataset_dir": str(dataset_dir), "gpu_request": GPU})
    shutil.copytree(Path(pipeline.__file__).parent, run_dir / "source", ignore=shutil.ignore_patterns("__pycache__"))
    try:
        training.main(argv, checkpoint_callback=volume.commit)
    finally:
        # A later Volume.reload() in this warm container requires closing open files.
        logger = logging.getLogger("cs336.train")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        volume.commit()
    print(f"Saved experiment to {run_dir}", flush=True)


@app.function(gpu=GPU, cpu=2, memory=8192, timeout=15 * 60, scaledown_window=2)
def generate(
    run_name: str, prompt: str = "Once upon a time", max_new_tokens: int = 256,
    temperature: float = 0.8, top_p: float = 0.9, seed: int = 42, checkpoint_step: int = 0,
):
    from cs336_basics.pipeline import generate_text

    volume.reload()
    result = generate_text(
        named_path("runs", run_name), prompt, max_new_tokens, temperature, top_p, seed, checkpoint_step,
    )
    volume.commit()
    return result


@app.function(cpu=1, memory=2048, timeout=5 * 60)
def plot(run_names: str, output: str = "comparison"):
    from cs336_basics.plot_metrics import main

    volume.reload()
    run_dirs = [str(named_path("runs", name.strip())) for name in run_names.split(",")]
    output_path = named_path("plots", output).with_suffix(".png")
    main([*run_dirs, "--output", str(output_path)])
    volume.commit()
    return str(output_path)

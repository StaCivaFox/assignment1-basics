"""CLI and setup for Section 5; the student implements train().

Run ``uv run python -m cs336_basics.train --help`` for all options.
Use --dry-run to check data and initialization without taking training steps.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from cs336_basics import data, nn_utils, optimizer as optim
from cs336_basics.model import TransformerLM


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    files = parser.add_argument_group("data and output")
    files.add_argument("--train-data", type=Path, required=True, help="Token IDs in a .npy file or raw binary file")
    files.add_argument("--val-data", type=Path, required=True, help="Held-out token IDs in the same supported formats")
    files.add_argument(
        "--token-dtype",
        choices=("uint16", "uint32", "int32", "int64"),
        help="Required for raw binary files; .npy files store their own dtype",
    )
    files.add_argument(
        "--run-dir", type=Path, help="Directory for config.json, train.log, metrics.jsonl, and checkpoints"
    )
    files.add_argument("--resume", type=Path, help="Checkpoint to restore using your data.load_checkpoint")

    model = parser.add_argument_group("model")
    model.add_argument("--vocab-size", type=int, required=True, help="Must match the tokenizer used for these arrays")
    model.add_argument("--d-model", type=int, default=128)
    model.add_argument("--num-layers", type=int, default=2)
    model.add_argument("--num-heads", type=int, default=4)
    model.add_argument("--d-ff", type=int, default=384)
    model.add_argument("--context-length", type=int, default=128)
    model.add_argument("--rope-theta", type=float, default=10000.0)

    training = parser.add_argument_group("training")
    training.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu, cuda:0, or mps")
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--steps", type=int, default=1000, help="Total completed updates, including restored updates")
    training.add_argument("--max-lr", type=float, default=1e-3)
    training.add_argument("--min-lr", type=float, default=1e-4)
    training.add_argument("--warmup-steps", type=int, default=100)
    training.add_argument("--anneal-steps", type=int, help="Final annealing iteration; defaults to --steps")
    training.add_argument("--beta1", type=float, default=0.9)
    training.add_argument("--beta2", type=float, default=0.999)
    training.add_argument("--eps", type=float, default=1e-8)
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument("--max-grad-norm", type=float, default=1.0)
    training.add_argument("--seed", type=int, default=42)

    reporting = parser.add_argument_group("reporting")
    reporting.add_argument("--log-every", type=int, default=10)
    reporting.add_argument("--eval-every", type=int, default=100)
    reporting.add_argument("--eval-batches", type=int, default=10)
    reporting.add_argument("--checkpoint-every", type=int, default=100)
    reporting.add_argument("--dry-run", action="store_true", help="Check setup and exit before calling train()")

    args = parser.parse_args(argv)
    if args.run_dir is None:
        args.run_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if args.anneal_steps is None:
        args.anneal_steps = args.steps
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "vocab_size",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
        "context_length",
        "batch_size",
        "steps",
        "log_every",
        "eval_every",
        "eval_batches",
        "checkpoint_every",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads")
    if (args.d_model // args.num_heads) % 2 != 0:
        raise ValueError("The per-head dimension must be even for RoPE")
    if not 0 <= args.warmup_steps < args.anneal_steps:
        raise ValueError("Require 0 <= warmup-steps < anneal-steps")
    for name in ("max_lr", "min_lr", "beta1", "beta2", "eps", "weight_decay", "max_grad_norm", "rope_theta"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if not 0 <= args.min_lr <= args.max_lr:
        raise ValueError("Require 0 <= min-lr <= max-lr")
    if not (0 <= args.beta1 < 1 and 0 <= args.beta2 < 1):
        raise ValueError("Both beta values must be in [0, 1)")
    if args.eps <= 0 or args.max_grad_norm <= 0 or args.rope_theta <= 0:
        raise ValueError("eps, max-grad-norm, and rope-theta must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be nonnegative")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be in [0, 2**32)")
    for path in (args.train_data, args.val_data, args.resume):
        if path is None:
            continue
        if not path.is_file():
            raise ValueError(f"File does not exist: {path}")
        if path != args.resume and path.suffix != ".npy" and args.token_dtype is None:
            raise ValueError("Specify --token-dtype for raw binary token files")


def setup_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cs336.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(run_dir / "train.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_metrics(logger: logging.Logger, run_dir: Path, iteration: int, **metrics: float) -> None:
    """Record scalar metrics supplied by your loop in the console, log file, and JSONL file."""
    values = {name: float(value) for name, value in metrics.items()}
    record = {"iteration": iteration, **values}
    line = json.dumps(record, allow_nan=False)
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    logger.info("iteration=%d %s", iteration, " ".join(f"{name}={value:.6g}" for name, value in values.items()))


def load_token_array(path: Path, token_dtype: str | None, context_length: int, vocab_size: int) -> np.ndarray:
    """Memory-map tokens; inspect a small prefix without scanning the entire corpus."""
    if path.suffix == ".npy":
        tokens = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        if token_dtype is None:
            raise ValueError("Raw token files require --token-dtype")
        tokens = np.memmap(path, mode="r", dtype=token_dtype)
    if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError(f"{path}: expected a one-dimensional integer array")
    if len(tokens) <= context_length:
        raise ValueError(f"{path}: need at least context_length + 1 tokens")
    prefix = tokens[:4096]
    if int(prefix.min()) < 0 or int(prefix.max()) >= vocab_size:
        raise ValueError(f"{path}: token IDs in the inspected prefix are outside [0, vocab_size)")
    return tokens


def train(
    args: argparse.Namespace,
    model: TransformerLM,
    optimizer: optim.AdamW,
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    start_iteration: int,
    logger: logging.Logger,
) -> None:
    """Student work: training updates, validation, and checkpoint/reporting cadence.

    Everything above and main() below is scaffolding. Implement the Section 5
    loop here, using your existing components in data.py, nn_utils.py, and
    optimizer.py. log_metrics() is available for scalar reporting.

    start_iteration counts completed updates. args.steps is the total target,
    including updates already completed before a checkpoint was saved.
    """
    # TODO: Implement your training loop here.
    for i in range(start_iteration, args.steps):
        # Get the learning rate for this iteration
        lr = optim.get_lr_cosine(i, args.max_lr, args.min_lr, args.warmup_steps, args.anneal_steps)
        # Sample a batch of input token IDs shaped (batch_size, context_length)
        token_ids_in, targets_in = data.get_batch(train_tokens, args.batch_size, args.context_length, args.device)
        # Clear previous gradients
        optimizer.zero_grad()
        # Forward
        logits = model(token_ids_in)
        # Compute cross-entropy loss
        loss = nn_utils.cross_entropy(logits, targets_in)
        # Backward
        loss.backward()
        # Clip the gradients
        nn_utils.gradient_clipping(model.parameters(), args.max_grad_norm)
        # Optimizer update
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        completed_run = i + 1
        # Run validation
        if completed_run % args.eval_every == 0:
            with torch.no_grad():
                eval_loss = torch.tensor(0.0, device=args.device)
                model.eval()
                for _ in range(args.eval_batches):
                    token_ids_eval, targets_eval = data.get_batch(val_tokens, args.batch_size, args.context_length, args.device)
                    logits_eval = model(token_ids_eval)
                    loss_eval = nn_utils.cross_entropy(logits_eval, targets_eval)
                    eval_loss += loss_eval
                eval_loss /= args.eval_batches
                model.train()
            log_metrics(logger, args.run_dir, completed_run, train_loss=loss.item(), val_loss=eval_loss.item(), lr=lr)
        # Logging
        if completed_run % args.log_every == 0:
            log_metrics(logger, args.run_dir, completed_run, train_loss=loss.item(), lr=lr)
        if completed_run % args.checkpoint_every == 0:
            data.save_checkpoint(model, optimizer, completed_run, args.run_dir / f"checkpoint{completed_run}")
    data.save_checkpoint(model, optimizer, args.steps, args.run_dir / f"checkpoint{args.steps}")


    


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = setup_logging(args.run_dir)
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    # Keep existing configs when writing additional logs into the same run directory.
    config_path = args.run_dir / "config.json"
    if config_path.exists():
        config_path = args.run_dir / f"config-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    logger.info("Configuration: %s", config_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_tokens = load_token_array(args.train_data, args.token_dtype, args.context_length, args.vocab_size)
    val_tokens = load_token_array(args.val_data, args.token_dtype, args.context_length, args.vocab_size)
    logger.info(
        "Train tokens=%d dtype=%s; validation tokens=%d dtype=%s",
        len(train_tokens),
        train_tokens.dtype,
        len(val_tokens),
        val_tokens.dtype,
    )

    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        rope_theta=args.rope_theta,
    ).to(device=args.device, dtype=torch.float32)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    start_iteration = 0
    if args.resume is not None:
        start_iteration = data.load_checkpoint(args.resume, model, optimizer)
        if not isinstance(start_iteration, int) or not 0 <= start_iteration <= args.steps:
            raise ValueError("Checkpoint iteration must be an integer between zero and --steps")
        logger.info("Restored checkpoint %s at %d completed updates", args.resume, start_iteration)
    logger.info(
        "Model parameters=%d; device=%s; completed updates=%d; target updates=%d",
        sum(p.numel() for p in model.parameters()),
        args.device,
        start_iteration,
        args.steps,
    )
    if args.dry_run:
        logger.info("Dry run complete. Data and initialization checked; train() was not called.")
        return
    train(args, model, optimizer, train_tokens, val_tokens, start_iteration, logger)


if __name__ == "__main__":
    main()

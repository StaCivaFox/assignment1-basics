import itertools
import json
import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from cs336_basics import train as training


def training_args(tmp_path, run_name="run", steps=7):
    for split in ("train", "val"):
        np.save(tmp_path / f"{split}.npy", np.arange(64, dtype=np.uint16) % 8)
    return [
        "--train-data", str(tmp_path / "train.npy"),
        "--val-data", str(tmp_path / "val.npy"),
        "--run-dir", str(tmp_path / run_name),
        "--vocab-size", "8", "--d-model", "8", "--num-heads", "2",
        "--d-ff", "16", "--num-layers", "1", "--context-length", "4",
        "--batch-size", "2", "--steps", str(steps), "--warmup-steps", "0",
        "--log-every", "2", "--eval-every", "3", "--eval-batches", "2",
        "--checkpoint-every", "5", "--experiment-name", "logging-smoke",
        "--notes", "Synthetic tokens; infrastructure test only.",
    ]


def read_metrics(run_dir):
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def test_training_logs_both_axes_without_duplicate_steps(tmp_path, monkeypatch):
    clock = itertools.count(100, 0.5)
    monkeypatch.setattr(training.time, "perf_counter", lambda: next(clock))
    training.main(training_args(tmp_path))
    run_dir = tmp_path / "run"
    records = read_metrics(run_dir)
    assert [record["iteration"] for record in records] == list(range(1, 8))
    assert [record["iteration"] for record in records if "val_loss" in record] == [3, 6, 7]
    times = [record["elapsed_seconds"] for record in records]
    assert times == sorted(times) and times[0] > 0
    assert all(record["train_loss"] > 0 and "lr" in record and "timestamp" in record for record in records)
    assert (run_dir / "checkpoint5").exists()
    assert (run_dir / "checkpoint7").exists()
    config = json.loads((run_dir / "config.json").read_text())
    assert config["experiment_name"] == "logging-smoke"
    assert config["notes"] == "Synthetic tokens; infrastructure test only."

    # Resume from an earlier checkpoint into a new experiment directory.
    # The time offset must come from step 5, not the later record at step 7.
    training.main(training_args(tmp_path, "resumed", steps=6) + ["--resume", str(run_dir / "checkpoint5")])
    resumed = read_metrics(tmp_path / "resumed")
    assert len(resumed) == 1
    assert resumed[0]["iteration"] == 6
    assert resumed[0]["elapsed_seconds"] == records[4]["elapsed_seconds"] + 0.5
    assert "val_loss" in resumed[0]


def test_log_metrics_records_nonfinite_losses_and_wandb_axes(tmp_path):
    run = MagicMock()
    training.log_metrics(
        logging.getLogger("test"), tmp_path, 10, elapsed_seconds=2.5,
        wandb_run=run, train_loss=float("nan"), val_loss=float("inf"), lr=0.001,
    )
    record = read_metrics(tmp_path)[0]
    assert record["train_loss"] is None
    assert record["val_loss"] is None
    assert record["nonfinite_metrics"] == ["train_loss", "val_loss"]
    values = run.log.call_args.args[0]
    assert values["iteration"] == 10
    assert values["elapsed_seconds"] == 2.5
    assert values["step/lr"] == values["time/lr"] == 0.001
    assert values["nonfinite_metric_count"] == 2


def test_setup_wandb_defines_both_curve_axes(tmp_path, monkeypatch):
    wandb = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    args = SimpleNamespace(
        wandb_project="test", experiment_name="lr-test", notes="Change learning rate",
        run_dir=tmp_path, wandb_mode="offline",
    )
    run = training.setup_wandb(args, {"max_lr": 0.001})
    assert run is wandb.init.return_value
    assert wandb.init.call_args.kwargs["mode"] == "offline"
    assert wandb.init.call_args.kwargs["config"] == {"max_lr": 0.001}
    run.define_metric.assert_any_call("step/*", step_metric="iteration")
    run.define_metric.assert_any_call("time/*", step_metric="elapsed_seconds")
    args.wandb_project = None
    assert training.setup_wandb(args, {}) is None
    assert wandb.init.call_count == 1


def test_old_checkpoint_without_timing_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert training.restore_elapsed_seconds(tmp_path / "checkpoint5", 5, logging.getLogger("test")) == 0
    assert "elapsed time starts at this session" in caplog.text


def test_dry_run_does_not_start_wandb_or_write_metrics(tmp_path, monkeypatch):
    setup = MagicMock()
    monkeypatch.setattr(training, "setup_wandb", setup)
    training.main(training_args(tmp_path) + ["--dry-run", "--wandb-project", "test"])
    setup.assert_not_called()
    assert not (tmp_path / "run" / "metrics.jsonl").exists()


@pytest.mark.parametrize("fails", [False, True])
def test_wandb_is_finished_on_success_and_failure(tmp_path, monkeypatch, fails):
    run = MagicMock()
    monkeypatch.setattr(training, "setup_wandb", lambda *args: run)
    loop = MagicMock(side_effect=RuntimeError("training failed") if fails else None)
    monkeypatch.setattr(training, "train", loop)
    if fails:
        with pytest.raises(RuntimeError, match="training failed"):
            training.main(training_args(tmp_path))
        run.finish.assert_called_once_with(exit_code=1)
    else:
        training.main(training_args(tmp_path))
        run.finish.assert_called_once_with()

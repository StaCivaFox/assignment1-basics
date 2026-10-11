import io
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from cs336_basics import pipeline, train


@pytest.mark.parametrize("chunk_chars", [1, 7, 128])
def test_document_stream_preserves_text_and_split_delimiters(chunk_chars):
    text = "A café.\n<|endoftext|>\nAnother story.<|endoftext|>\n"
    documents = list(pipeline.iter_documents(io.StringIO(text), chunk_chars))
    assert documents == ["A café.\n<|endoftext|>", "\nAnother story.<|endoftext|>", "\n"]
    assert "".join(documents) == text


def test_download_uses_separate_splits_and_reuses_matching_inputs(tmp_path, monkeypatch):
    sources = {
        pipeline.SOURCES["train"]: b"Training story.<|endoftext|>\nSecond.<|endoftext|>",
        pipeline.SOURCES["valid"]: b"Held-out story.<|endoftext|>\n",
    }
    opener = MagicMock(side_effect=lambda url, **kwargs: io.BytesIO(sources[url]))
    monkeypatch.setattr(pipeline, "urlopen", opener)
    pipeline.download_dataset(tmp_path, train_stories=1, val_stories=0)
    assert (tmp_path / "train.txt").read_text() == "Training story.<|endoftext|>"
    assert (tmp_path / "valid.txt").read_text() == "Held-out story.<|endoftext|>\n"
    pipeline.download_dataset(tmp_path, train_stories=1, val_stories=0)
    assert opener.call_count == 2
    with pytest.raises(ValueError, match="new dataset name"):
        pipeline.download_dataset(tmp_path, train_stories=2, val_stories=0)


def test_bpe_tokenization_training_and_generation(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    training_text = (
        "Once upon a time, a little girl played with a happy dog.\n<|endoftext|>\n"
        "The kind boy found a bright red ball in the garden.\n<|endoftext|>\n"
    ) * 4
    validation_text = "A friendly cat watched the children play outside.\n<|endoftext|>\n" * 2
    (dataset_dir / "train.txt").write_text(training_text)
    (dataset_dir / "valid.txt").write_text(validation_text)
    pipeline.fit_bpe(dataset_dir, vocab_size=280, bpe_stories=2)
    # Tokenization must preserve full-string encoding even with small batches.
    tokenizer = pipeline.load_tokenizer(dataset_dir)
    ids_path = tmp_path / "streamed.bin"
    count = pipeline.encode_file(tokenizer, dataset_dir / "train.txt", ids_path, batch_chars=20)
    ids = np.fromfile(ids_path, dtype=np.uint16).tolist()
    assert ids == tokenizer.encode(training_text)
    assert count == len(ids)
    assert tokenizer.decode(ids) == training_text
    metadata = pipeline.tokenize_dataset(dataset_dir)
    assert metadata["vocab_size"] == 280
    assert metadata["token_counts"]["train"] == count
    assert pipeline.tokenize_dataset(dataset_dir) == metadata

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pipeline.write_json(run_dir / "pipeline.json", {"dataset_dir": str(dataset_dir)})
    extra_args = [
        "--steps", "3", "--batch-size", "2", "--d-model", "16", "--num-layers", "1",
        "--num-heads", "2", "--d-ff", "32", "--context-length", "16", "--warmup-steps", "0",
        "--log-every", "1", "--eval-every", "2", "--eval-batches", "1", "--checkpoint-every", "2",
    ]
    committed_steps = []

    def commit_checkpoint():
        records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
        step = records[-1]["iteration"]
        assert (run_dir / f"checkpoint{step}").is_file()
        committed_steps.append(step)

    train.main(
        pipeline.training_arguments(dataset_dir, run_dir, extra_args, device="cpu"),
        checkpoint_callback=commit_checkpoint,
    )
    assert committed_steps == [2, 3]
    result = pipeline.generate_text(run_dir, "Once upon a time", max_new_tokens=8, device="cpu")
    assert result["checkpoint"] == "checkpoint3"
    assert 1 <= result["generated_token_count"] <= 8
    assert result["text"].startswith("Once upon a time")
    assert len(list(run_dir.glob("generation-*.txt"))) == 1
    assert json.loads(next(run_dir.glob("generation-*.json")).read_text()) == result


def test_pipeline_disallows_overriding_dataset_paths(tmp_path):
    with pytest.raises(ValueError, match="Pipeline sets"):
        pipeline.training_arguments(tmp_path, tmp_path / "run", ["--vocab-size=1024"])


def test_requesting_too_many_stories_reports_error(tmp_path):
    with pytest.raises(ValueError, match="found 1"):
        pipeline.copy_stories(io.StringIO("One.<|endoftext|>"), tmp_path / "sample.txt", 2)

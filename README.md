# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Track experiments (Section 7.1)

Train on token arrays produced by your tokenizer, using a separate run directory
for each experiment:

```sh
uv run python -m cs336_basics.train \
  --train-data data/train.npy --val-data data/val.npy --vocab-size 10000 \
  --run-dir runs/lr-1e-3 --experiment-name lr-1e-3 \
  --notes "Section 7.2: baseline learning-rate trial" \
  --max-lr 1e-3 --log-every 10 --eval-every 100 --eval-batches 10
```

Each run saves its hyperparameters, seed, data paths, name, and notes in
`config.json`, readable progress in `train.log`, and scalar records in
`metrics.jsonl`. Each metric record contains:

- `iteration`: number of completed optimizer updates.
- `elapsed_seconds`: elapsed training-loop wall-clock time, including validation,
  earlier logging, and checkpointing; excludes initialization and pauses between sessions.
- `train_loss`: cross-entropy on the latest training minibatch, before its update.
- `val_loss`: mean cross-entropy over `--eval-batches`, when validation runs.
- `lr`: learning rate used for this update, and `timestamp`: UTC recording time.

The first update, reporting intervals, validation events, checkpoints, and final
update are recorded. Coinciding events share one record. Validation also runs on
the final update. Nonfinite metrics are stored as JSON `null` and listed in
`nonfinite_metrics`, so divergent experiments leave readable evidence.

Export local loss curves against both steps and seconds, optionally comparing runs:

```sh
uv run --with matplotlib python -m cs336_basics.plot_metrics \
  runs/lr-1e-3 runs/lr-3e-4 --output runs/learning-rate-comparison.png
```

For optional Weights & Biases charts, add `--wandb-project cs336-assignment1` to
the training command. The `step/` charts use completed updates and the `time/`
charts use elapsed seconds, configured through
[W&B custom metric axes](https://docs.wandb.ai/ref/python/experiments/run/).
Use `--wandb-mode offline` to store W&B data locally for later synchronization;
online mode requires your W&B account to be configured. Without `--wandb-project`,
training uses only the local logs.

To continue training, supply `--resume runs/lr-1e-3/checkpoint1000` and a larger
`--steps` value. Keep `metrics.jsonl` alongside the source checkpoint: elapsed time
is restored from the record at that checkpoint's step, including when the new
session uses a different run directory. Older checkpoints without timing records
start their timer at zero with a warning. Each invocation creates a new W&B run
when W&B is enabled; use a new local directory when branching from an older checkpoint.

For the written experiment log required by Section 7.1, record each run's path,
what changed, why you tried it, validation result or divergence, curve filename,
and your interpretation. `--notes` captures the initial purpose; add your findings
to your assignment write-up after inspecting the results. Synthetic smoke tests
only check the infrastructure and are not TinyStories experiment results.

### Run the TinyStories pipeline on Modal

[`modal_app.py`](./modal_app.py) wraps your existing implementations with six
remote commands. Run the commands below from this repository's root directory.
The container uses Python 3.12, the dependencies in `uv.lock`, and the current
`cs336_basics` source, following Modal's
[Image and source packaging APIs](https://modal.com/docs/sdk/py/latest/Image).

| Stage | Resource | Persistent outputs |
| --- | --- | --- |
| `download` | CPU | Separate official training and validation text |
| `bpe` | CPU, 32 GiB RAM | `vocab.pkl`, `merges.pkl`, tokenizer settings |
| `tokenize` | CPU | Streamed `uint16` token arrays and token counts |
| `train` | One GPU; L40S by default | Config, logs, checkpoints, a copy of your source |
| `generate` | One GPU | Generated text, token IDs, decoding settings |
| `plot` | CPU | Loss curves against steps and wall-clock seconds |

All artifacts live in the persistent Volume `cs336-section7`, mounted at `/data`.
Each stage reloads the Volume to see preceding stages' writes and commits its
outputs. Training also commits after each completed checkpoint and metric record.
See Modal's [Volume documentation](https://modal.com/docs/guide/volumes).

**1. Install the CLI and authenticate once.**

```sh
uv tool install 'modal[api-proxy-support]>=1.6,<2'
modal setup
```

The `api-proxy-support` extra installs the dependencies Modal needs when proxy
environment variables such as `HTTPS_PROXY` are set, including in WSL2. If Modal
is already installed without this extra, rerun the install command with `--force`.

Create a Modal account if needed. GPU usage requires a payment method; setup is
described in the [Modal quickstart](https://modal.com/docs/guide). These commands
start remote compute when you run them. See [current pricing](https://modal.com/pricing)
for CPU, GPU, memory, and storage charges.

**2. Check the complete pipeline with a small dataset and model.**

Run these commands in order, waiting for each stage to finish:

```sh
modal run modal_app.py::download --dataset smoke --train-stories 1000 --val-stories 100
modal run modal_app.py::bpe --dataset smoke --vocab-size 512
modal run modal_app.py::tokenize --dataset smoke

modal run modal_app.py::train --dataset smoke --run-name smoke-001 \
  --train-args "--d-model 64 --num-layers 2 --num-heads 4 --d-ff 128 \
    --context-length 64 --batch-size 4 --steps 20 --warmup-steps 2 \
    --eval-every 10 --eval-batches 2 --checkpoint-every 10"

modal run modal_app.py::generate --run-name smoke-001 \
  --prompt "Once upon a time" --max-new-tokens 256 --temperature 0.8 --top-p 0.9
```

This checks the connections between the components; 20 updates on this small
sample are not expected to produce fluent stories. `generate` loads the latest
checkpoint by default, looks up the EOS ID from your vocabulary, and stops at EOS
or the token limit. Use `--checkpoint-step 10` to select an earlier checkpoint.

**3. Prepare the full dataset once for Section 7 experiments.**

```sh
modal run --detach modal_app.py::download --dataset tinystories-10k \
  --train-stories 0 --val-stories 0
# Wait for download to finish before starting BPE.
modal run --detach modal_app.py::bpe --dataset tinystories-10k --vocab-size 10000
# Wait for BPE to finish before tokenization.
modal run --detach modal_app.py::tokenize --dataset tinystories-10k
```

Zero story limits mean the full corresponding official split. BPE is fitted only
on training text; validation uses that same tokenizer. Tokenization writes raw
`train.bin` and `valid.bin` files in batches ending at EOS, which your training CLI
reads using `np.memmap`. The existing BPE and tokenization algorithms are serial;
these CPU stages may take time, and the resulting artifacts can be reused across
learning-rate, batch-size, and architecture experiments.

For a faster preliminary tokenizer, `bpe --bpe-stories 100000` trains on the first
100,000 training stories; record this choice in your experiment notes. Use a new
dataset name for a different vocabulary or preparation, since completed stages
are cached and their configuration is checked before reuse.

`--detach` keeps a remote function running after the client disconnects. Follow
the app URL printed by Modal to check completion before launching the next stage.
See [Modal app lifecycle](https://modal.com/docs/guide/apps). The configured limits
are 4 hours for downloading/training, 24 hours for BPE/tokenization, and 15 minutes
for generation; adjust the decorators in `modal_app.py` if needed, up to Modal's
[24-hour function timeout](https://modal.com/docs/guide/timeouts).

**4. Train a baseline, then vary one setting per experiment.**

The pipeline defaults to the Section 7 dimensions: vocabulary from the prepared
dataset, context 256, `d_model=512`, `d_ff=1344`, 4 layers, 16 heads, RoPE theta
10,000. The initial learning rate is a trial value to tune, not a validated optimum.

```sh
# A first run using the handout's reduced token budget:
# 32 examples/update × 256 tokens/example × 5000 updates = 40,960,000 tokens.
modal run --detach modal_app.py::train --dataset tinystories-10k --run-name lr-1e-3 \
  --train-args "--batch-size 32 --steps 5000 --max-lr 0.001 --min-lr 0.0001 \
    --warmup-steps 100 --notes 'Reduced-budget learning-rate baseline'"

# Full Section 7 token budget on one B200:
# 64 × 256 × 20000 = 327,680,000 tokens.
CS336_MODAL_GPU=B200 modal run --detach modal_app.py::train \
  --dataset tinystories-10k --run-name full-lr-1e-3 \
  --train-args "--batch-size 64 --steps 20000 --max-lr 0.001 --min-lr 0.0001 \
    --warmup-steps 200 --notes 'Full-token-budget baseline'"
```

Choose one budget for a comparison and keep total training tokens comparable.
For example, repeat the first command with `--run-name lr-3e-4`,
`--max-lr 0.0003`, and `--min-lr 0.00003`. Run names must be unique. All ordinary
training hyperparameters are passed through `--train-args`; dataset paths,
vocabulary size, output path, and device are supplied by the wrapper.
For architecture ablations, edit your model code and start a new run; each run
saves its source under `source/`. Generate with the matching architecture code.

`CS336_MODAL_GPU` selects the GPU for training and generation. `L40S` and `B200`
are supported [Modal GPU types](https://modal.com/docs/guide/gpu). Measure your
implementation's runtime with the smoke run before choosing a long budget; the
handout's reference runtime is not a guarantee for this implementation.

If a run times out or is interrupted, list its checkpoints and resume into a new
run directory using the same model dimensions and learning-rate schedule:

```sh
modal volume ls cs336-section7 runs/lr-1e-3
modal run --detach modal_app.py::train --dataset tinystories-10k --run-name lr-1e-3-resumed \
  --train-args "--resume /data/runs/lr-1e-3/checkpoint2000 --steps 5000 \
    --batch-size 32 --max-lr 0.001 --min-lr 0.0001 --warmup-steps 100"
```

Replace `checkpoint2000` with a completed checkpoint that actually exists. This
restores weights, optimizer state, update count, and logged elapsed time. The
current checkpoint format does not restore the RNG state, so the resumed sample
sequence is not identical to an uninterrupted run.

**5. Generate text and collect the Section 7 deliverables.**

```sh
modal run modal_app.py::generate --run-name lr-1e-3 \
  --prompt "Once upon a time" --max-new-tokens 256 --temperature 0.8 --top-p 0.9
modal run modal_app.py::plot --run-names lr-1e-3,lr-3e-4 --output lr-comparison

modal volume get cs336-section7 runs/lr-1e-3 runs/modal-lr-1e-3
modal volume get cs336-section7 plots/lr-comparison.png runs/lr-comparison.png
```

The downloaded run contains `config.json`, `metrics.jsonl`, `train.log`,
`checkpoint*`, `generation-*.txt`, `generation-*.json`, and `source/`.
Download `datasets/tinystories-10k/vocab.pkl` and `merges.pkl` as well if you want
to use the tokenizer locally. The [Volume CLI](https://modal.com/docs/cli/latest/volume)
supports recursive directory downloads. Update your experiment log with the run's
purpose, final validation loss, curve, generated sample, and findings.

Local JSONL logs and exported plots work without W&B. To retain an offline W&B
run too, add `--wandb-project cs336-assignment1 --wandb-mode offline` inside
`--train-args`; its files are saved under the run's `wandb/` directory.

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```


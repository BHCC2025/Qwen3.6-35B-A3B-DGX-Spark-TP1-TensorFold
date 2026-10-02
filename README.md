# Qwen3.6-35B-A3B on one DGX Spark (TP1), TensorFold

Built on [TensorFold](https://github.com/ashhart/TensorFold) by @ashhart. This recipe installs TensorFold v0.6.2
from its upstream repository, unmodified, and runs `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP` with it on one NVIDIA DGX
Spark (GB10). One script, `./run.sh tp1`, and one config file. It fits a **64 GB Spark** as well as a 128 GB one.

TensorFold serves the 35B-A3B mixture-of-experts model (3B active parameters a token) from its MLX 4-bit checkpoint and
drafts with the model's own MTP layer, so there is no separate draft model to download. With `--parallel 8` it decodes
up to eight requests in shared rounds, and each reply is token for token the same as that request served alone: the
bench checks it for every concurrent greedy request (54 of 54 equal). The serve command is the one in TensorFold's
[Qwen3.6 recipe](https://github.com/ashhart/TensorFold/blob/56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21/docs/recipes/qwen3.6-moe.md);
this repository adds the setup, a memory profile for 64 GB Sparks and the benchmark around it.

| Spark | Command | Context | Decode, single stream (code / prose) | 8 users, total (code / chat) | Cold prefill 8K | Verified |
|---|---|---|---|---|---|---|
| 128 GB | `./run.sh tp1` | auto: up to 262,144 tokens × 8 streams | 160.5 / 133.3 tok/s | 558.5 / 471.7 tok/s | 7,112 tok/s | 2026-10-02 |
| 64 GB | `./run.sh tp1` (picks `MEM_GB=64`) | 65,536 tokens × 4 streams | 159.9 / 132.8 tok/s | 414.6 / 344.7 tok/s | 7,055 tok/s | 2026-10-02, simulated 64 GB |

Every row is benched on our own Spark with `bench/bench.sh` (same prompts for every row) and must pass the smoke
test before it is marked verified. We have no 64 GB Spark: that row ran on a 128 GB one with all but 53.5 GiB locked
away (see [Memory](#memory-and-64-gb-sparks)). See [bench/results/](bench/results/).

- **Endpoint:** `http://<spark>:8000/v1` (OpenAI-compatible), model `qwen3.6-35b-a3b`
- **Defaults:** MTP drafting on; 8 concurrent requests (`PARALLEL=8`; 4 on a 64 GB Spark), more wait their turn;
  context sized by TensorFold to the memory it can afford (65,536 tokens on a 64 GB Spark); thinking on unless a
  request sends `chat_template_kwargs: {"enable_thinking": false}`; tool calls; sampling from the model's generation
  config (temperature 1.0, top_k 20, top_p 0.95) unless the request sets it.

## Requirements

| | |
|---|---|
| Hardware | 1 DGX Spark, 128 GB or 64 GB (or another GB10 box) |
| OS | DGX OS 7 (Ubuntu 24.04), Docker with the NVIDIA runtime |
| Disk | 25 GB free NVMe for the model, plus ~25 GB for the image |
| Engine | TensorFold v0.6.2 (`56e2e3e`), built into `dgx-spark-tensorfold:v0.6.2` from [docker/Dockerfile](docker/Dockerfile) on `nvcr.io/nvidia/pytorch:26.07-py3` |
| Model | `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP` @ `f84b054c` (formerly `Vontra/Qwen3.6-35B-A3B-MLX-4bit-MTP`; 20.9 GB, MTP layer included) |
| Access | `sudo` for installs |

## Quick start

Before you start:
- DGX OS 7 on the Spark, with its current updates.
- Nothing to set up on Hugging Face unless the model is gated: `./setup.sh` test-downloads one small file first, and
  if the model is gated it shows the licence page to accept and offers to log you in.

On the Spark:

```bash
git clone https://github.com/BHCC2025/Qwen3.6-35B-A3B-DGX-Spark-TP1-TensorFold.git
cd Qwen3.6-35B-A3B-DGX-Spark-TP1-TensorFold
./setup.sh
```

`setup.sh` asks how many Sparks you have (answer 1), then:
- checks and installs what's missing
- writes `cluster.env` for you
- builds the image (NVIDIA's PyTorch container + TensorFold from upstream) and checks the GPU is visible inside it
- downloads the model

It asks before every change. Re-run it any time. `./setup.sh --check` only reports.

Then start it:

```bash
./run.sh tp1        # one Spark; on a 64 GB Spark it picks the 64 GB profile by itself
./run.sh status     # wait for "serving: [...]"; the first start compiles TensorFold's kernels (about 2 minutes)
scripts/smoke-test.sh
```

Stop with `./run.sh stop`.

## Settings

Set any of these in the environment for one run (`PARALLEL=4 ./run.sh tp1`), or put them in `cluster.env` to keep
them. `DRY_RUN=1` prints the docker command and starts nothing.

| Variable | 128 GB | 64 GB | What it does |
|---|---|---|---|
| `MEM_GB` | auto | auto | Memory profile: `auto` reads `/proc/meminfo` (under 80 GiB means 64), or `128` / `64` |
| `PARALLEL` | 8 | 4 | Requests decoded together (`--parallel`); more wait their turn |
| `CONTEXT` | auto | 65536 | `--context`, prompt plus reply per stream; `auto` lets TensorFold size it to the memory it can afford |
| `KV_DTYPE` | bf16 | bf16 | KV cache: `bf16`, `int8`, `int4` |
| `DRAFTS` | 1 | 1 | `0` = `--no-drafts`, TensorFold's serial reference (same replies, slower) |
| `EXTRA` / `DOCKER_EXTRA` | | | extra args for `tensorfold serve` / `docker run` |

`PARALLEL` and `CONTEXT` set explicitly win over the profile. `IMAGE` can be set the same way, and so can any
`cluster.env` value (`PORT=8001 ./run.sh tp1`).

## How it works

- **The image.** TensorFold has no published container. `docker/Dockerfile` takes NVIDIA's PyTorch container
  (`nvcr.io/nvidia/pytorch:26.07-py3`, pinned by digest in `recipe.yaml`), clones TensorFold at the v0.6.2 commit and
  `pip install`s it, which adds TensorFold alone and leaves the container's torch, Triton and CUDA as they are (the way
  TensorFold's README installs it). It is the same image as the other TensorFold recipes here. Nothing in TensorFold
  is patched.
- **Checkpoint.** The four weight shards are mlx-community's MLX 4-bit conversion (affine, groups of 64, routers and
  the shared-expert gate at 8 bits); `mtp-4bit.safetensors` beside them adds the model's MTP layer, converted the same
  way. TensorFold reads the family (`qwen3_5_moe`) from `config.json`, so no model flags are needed.
- **Drafting.** Each round TensorFold drafts a chain of up to three tokens with the MTP layer (or copies a repeated
  stretch of the context) and verifies them in one forward pass against its own serial sample, so drafted output
  equals `DRAFTS=0` output. A request decoded alongside others gives the same tokens as alone;
  `CONCURRENT=1,8 bench/bench.sh` checks it.
- **Concurrency.** A lone request replays one-stream CUDA graphs, so `PARALLEL=8` costs a single user nothing
  (160.5 tok/s against 159.1 with `PARALLEL=1`); with eight users it serves 2.7× the total of `PARALLEL=1`, where
  they queue.

### Memory and 64 GB Sparks

The weights take 19.5 GiB. At startup TensorFold estimates the memory for every stream filled to its window plus
three kept prompt states, and refuses a `CONTEXT` that does not fit; the caches then fill as contexts grow.

| Profile | Startup estimate | Measured peak of the server (bench + every stream filled to ~63K tokens) |
|---|---:|---:|
| 128 GB: `PARALLEL=8`, `CONTEXT=auto` (237,329-262,144 tokens) | 88.7-95.6 GiB | 33.9 GiB system-wide (8 streams × ~63K), so far from its estimate |
| 64 GB: `PARALLEL=4`, `CONTEXT=65536` | 34.7 GiB | 38.1 GiB system-wide, 27.8 GiB on the GPU (4 streams × ~63K) |

On a 64 GB Spark `CONTEXT=auto` with 8 streams would size the window to TensorFold's whole budget, about 47 GiB,
leaving only its own reserve (a tenth of memory) for the OS. The 64 GB profile caps the estimate at 34.7 GiB instead.
We have no 64 GB Spark, so we simulated one: a 128 GB Spark with memory locked away (`mlock`) until 53.5 GiB was
available (a 64 GB Spark has about 57.5 GiB, less about 4 GiB for the OS), TensorFold told the reserve it would
compute there (`TENSORFOLD_MEMORY_RESERVE_GIB=5.8`), and the full bench plus four streams filled to ~63K tokens. It
loaded, passed and kept 14.8 GiB available at its lowest. Other profiles that fit, by TensorFold's estimate:
`PARALLEL=8 CONTEXT=32768` (31.8 GiB), `PARALLEL=1 CONTEXT=131072` (32.1 GiB) or `PARALLEL=2 CONTEXT=131072`
(41.4 GiB); those were not benched. Details in [bench/results/2026-10-02-tp1.md](bench/results/2026-10-02-tp1.md).

## Benchmarks

`bench/bench.sh LABEL` runs the same suite against whatever is serving on `:8000` (the shared suite from the kit, so
every recipe is measured the same way):
- single-stream decode for code, prose and a ~9K-token prompt
- cold prefill at 8K and 28K tokens (unique prompts, so no prefix-cache hits)
- the smoke test; add `LONG=1` for the needle test
- with `CONCURRENT=1,8`: TensorFold's own `tools/bench_concurrent.py` (from the same commit, inside the container) at
  1 and 8 users, greedy, each concurrent reply checked against the same request alone

Results and raw logs go in [bench/results/](bench/results/). The TP1 runs (2026-10-02, one Spark):

| Test | Default (`PARALLEL=8`) | `PARALLEL=1` | 64 GB profile, simulated |
|---|---:|---:|---:|
| Short code / prose decode, single stream (median of 3) | 160.5 / 133.3 tok/s | 159.1 / 132.6 tok/s | 159.9 / 132.8 tok/s |
| ~9.3K-token prompt, decode (cold run) | 113.2 tok/s | 112.4 tok/s | 108.7 tok/s |
| Cold prefill ~7.3K / ~25.7K tokens | 7,112 / 6,288 tok/s | 6,941 / 6,187 tok/s | 7,055 / 6,258 tok/s |
| 1 user, greedy, code / chat | 214.4 / 164.2 tok/s | 211.7 / 159.9 tok/s | 214.6 / 162.3 tok/s |
| 8 users, greedy, code / chat: total | 558.5 / 471.7 tok/s | 207.9 / 159.4 tok/s (queued) | 414.6 / 344.7 tok/s (4 at a time) |
| 8 users, greedy, code / chat: per user | 75.8 / 62.4 tok/s | | 106.7 / 87.8 tok/s |
| Concurrent replies equal to the request alone | 54 / 54 | 54 / 54 | 54 / 54 |
| Smoke test | all pass | all pass | all pass |

The needle test (`LONG=1`) does not run against TensorFold yet: it sizes its prompts with vLLM's `/tokenize`
endpoint. For TensorFold's own measurements of this model on a Spark (and against vLLM), see its
[Qwen3.6 recipe](https://github.com/ashhart/TensorFold/blob/56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21/docs/recipes/qwen3.6-moe.md#measurements).

## Troubleshooting

Run `./setup.sh --check` and read the FAIL lines. It writes `.setup/report.txt`, which is what to attach to an issue.
See also [docs/troubleshooting.md](docs/troubleshooting.md). The most common problems:
- Empty `content` with a long `reasoning_content`: thinking is on by default and Qwen3.6 thinks at length, so a small
  `max_tokens` ends inside the think block. Raise `max_tokens`, or send `chat_template_kwargs: {"enable_thinking": false}`
  (the smoke test does this for its direct-answer checks, with the kit's `SMOKE_NO_THINK=1`).
- `CUDA startup memory budget cannot fit ...` at start: the `CONTEXT` asked for does not fit the memory free right
  now. Stop other containers, or lower `PARALLEL` / `CONTEXT`; the message gives the window that fits.
- Out of memory while loading: other containers are still running, or page cache is taking memory. Run `./run.sh stop` and check `free -g`.

## Credits

Built on [TensorFold](https://github.com/ashhart/TensorFold) by @ashhart (Apache-2.0), installed from upstream at
v0.6.2 and not modified. Bug reports about the engine belong in its repository. The serve settings follow TensorFold's
Qwen3.6 recipe (`docs/recipes/qwen3.6-moe.md`); the concurrency test is TensorFold's own `tools/bench_concurrent.py`.

- Model: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) by the Qwen team (Apache-2.0).
- Checkpoint: [TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP](https://huggingface.co/TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP)
  (published as Vontra/Qwen3.6-35B-A3B-MLX-4bit-MTP): the weight shards are
  [mlx-community/Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit), the MTP layer is
  converted from Qwen's release by the checkpoint's publisher.
- Container base: NVIDIA's PyTorch container, under NVIDIA's licence terms (printed when it starts).

Full details are in [NOTICE](NOTICE).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

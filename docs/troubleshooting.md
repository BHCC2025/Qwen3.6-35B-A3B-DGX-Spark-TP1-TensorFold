# Troubleshooting

Start with `./setup.sh --check` (it changes nothing), then look at the server log (`./run.sh logs`).

| Symptom | Cause | Fix |
|---|---|---|
| Download fails with 401/403 | Gated model: licence not accepted, or not logged in | Accept the licence on the model's Hugging Face page, then `hf auth login` |
| `MODEL MISSING at ...` | The model (or the draft model) isn't on that node | `./setup.sh` (downloads it and copies it to every node) |
| `image missing` / `Unable to find image` | The image is built locally, never pulled | `./setup.sh` builds it from `docker/Dockerfile` on the head and copies it to the workers |
| The first start takes several minutes | TensorFold compiles its CUDA kernels on first use | Wait; they are kept in `CACHE_DIR` (per node), so later starts are quick |
| OOM or a Spark reboots while loading | Other containers are holding memory; page cache | Stop everything else, then check `free -g` |
| Context smaller than expected | `CONTEXT=auto` sizes the window to the memory left after `PARALLEL` streams | Lower `PARALLEL`, or set `CONTEXT` explicitly (TensorFold refuses one that cannot fit) |
| A setting from the environment is ignored | It isn't one of the knobs the launchers read | The Settings table in the README lists them; a new knob also needs adding to `FORWARD_VARS` in `lib/common.sh` so the workers get it |
| Empty `content`, long `reasoning_content` | Thinking is on by default and Qwen3.6 thinks at length; `max_tokens` ran out inside the think block | Raise `max_tokens`, or send `chat_template_kwargs: {"enable_thinking": false}` |
| `CUDA startup memory budget cannot fit ...` | The `CONTEXT` asked for (or the 64 GB profile's 65,536) does not fit the memory free right now | Stop other containers and check `free -g`; or lower `PARALLEL` / `CONTEXT` (the message gives the window that fits) |
| A 64 GB Spark runs short of memory | `PARALLEL` or `CONTEXT` set above the 64 GB profile, or `MEM_GB=128` forced | Leave `MEM_GB=auto` (or set `MEM_GB=64`) and unset `PARALLEL` / `CONTEXT`; `./run.sh tp1` prints the profile it used |
| `drafts with its own MTP layer on CUDA: a separate draft model does not apply` | A `--drafter` was passed in `EXTRA`, or `DRAFT_DIR` set | Remove it: this model drafts with the MTP layer in its checkpoint |
| `mtp-4bit.safetensors` missing | The model folder is mlx-community's conversion without the MTP file | Point `MODEL_DIR` at the TensorFold checkpoint, which has it (`./setup.sh` downloads it) |

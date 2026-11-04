# Running the experiments on Tinker

The repo's experiments are written for PyTRIO. Tinker (Thinking Machines) has nearly the same API: PyTRIO's interface is modelled on it. This folder contains:

- **`grpo_tinker.py`**: a working Tinker port of `experiments/grpo/03-grpo.py` (GSM8K GRPO, Phase 2 of the study guide), with `importance_sampling`, `ppo` and `cispo` losses.
- A cheat sheet for porting any other chapter (below).

Checked against `tinker==0.37.0`. The script was tested offline with a mocked backend that uses the real Tinker types. It has **not** been run against the live service, so do the smoke test first.

## Quick start

```bash
export TINKER_API_KEY=...   # or: tinker auth login
```

```bash
# 1. see which models your account can train
uv run --with tinker python study-guide/tinker/grpo_tinker.py --list-models
```

```bash
# 2. smoke test: costs very little (2 steps × 2 prompts × 4 samples)
uv run --with tinker python study-guide/tinker/grpo_tinker.py \
    --steps 2 --batch-size 2 --group-size 4 --max-tokens 256 --swanlab-mode disabled
```

```bash
# 3. the Phase 2 experiment: repeat with --loss-fn ppo and --loss-fn cispo
uv run --with tinker python study-guide/tinker/grpo_tinker.py \
    --steps 10 --batch-size 4 --group-size 8 --max-tokens 512 \
    --loss-fn importance_sampling --swanlab-mode disabled
```

The default model is `Qwen/Qwen3-4B-Instruct-2507`, the closest Tinker model I know of to the repo's `Qwen/Qwen3.5-4B`. If `--list-models` doesn't show it, pass another one with `--base-model`. To get SwanLab charts, run `swanlab login` and use `--swanlab-mode online`.

## Porting cheat sheet (for the other chapters)

| PyTRIO (repo code) | Tinker | Notes |
|---|---|---|
| `import pytrio as trio` | `import tinker` (then `trio.X` → `tinker.X`) | `Datum`, `ModelInput.from_ints`, `SamplingParams`, `AdamParams`, `TensorData` all exist with the same fields |
| `trio login` | `export TINKER_API_KEY=...` or `tinker auth login` | |
| `trio.configure(...)`, `trio.cn` | delete | PyTRIO-only region/timeout settings |
| `sample(..., return_text=True)` | `sample(...)` | no `return_text`: decode with `tokenizer.decode(seq.tokens, skip_special_tokens=True)` |
| `sequence.text` | not available | same as above |
| `sequence.tokens`, `sequence.logprobs` | same | |
| `forward_backward(datums, loss_fn="importance_sampling" / "ppo")` | same | Tinker also has `cispo`, `gspo`, `dro` and more built in, plus `loss_fn_config=` for clip thresholds |
| `forward_backward_custom(datums, fn)` | same | used by GSPO / DAPO custom losses |
| `optim_step(adam_params)` | same | |
| `save_weights_and_get_sampling_client()` | same | |
| `save_weights_for_sampler(name)`, `save_state(name)` | same | paths are `tinker://...` instead of `trio://...`; use them in `eval.py --model-path` |
| `service_client.create_sampling_client(base_model=...)` | same | OPD teacher sampling / logprob scoring |
| `--base-model Qwen/Qwen3.5-4B`, teacher `Qwen/Qwen3.6-27B` | a model from `--list-models` | e.g. a larger Qwen3 model as the OPD teacher. Teacher and student must share a tokenizer. |
| `trio.ImageChunk` (Vision GRPO, Spec-o3) | check `--list-models` for a vision model | the image-input API may differ; port these last |

Suggested porting order, from easiest to hardest: `grpo` (done) → `general-opd` → `dapo` / `gspo` → `search-r1` / `retool` → `alfworld` / `agentopsd` / `tempo` → vision chapters.

**Chat templates:** the multi-turn chapters (`search-r1/protocol.py`, `retool/protocol.py`) depend on the exact Qwen3.5 chat template. With a different model family or version, rerun one trajectory and check that the "prefix extension" checks in `build_next_prompt` / `build_datum` don't raise errors.

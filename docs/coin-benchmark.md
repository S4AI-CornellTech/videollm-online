# COIN benchmark: state, findings, and how to resume

Working notes for measuring VideoLLM-online accuracy on the COIN benchmark.
Everything below was measured on the `s4ai` GB10 box unless marked as an estimate.

Last updated: 2026-10-06. Branch: `coin-benchmark`.

---

## 1. The headline finding

**The released checkpoint is not COIN-finetuned, and scores 5.0% where the paper reports 92.7%.**

`chenjoya/videollm-online-8b-v1plus` was trained on Ego4D GoalStep, not COIN. The paper
says so directly (Section 4.2):

> "Due to potential privacy risks associated with COIN dataset collected from YouTube
> videos, we have opted to use Ego4D Goal-Step for training our released model."

The paper's COIN numbers (Table 2a) come from a *separate* training run that was never
released — `scripts/coin/live1+.sh` is the recipe for reproducing it. Section 4.1 adds
that those runs train "5~6 epochs **without pre-training**", i.e. starting from
Llama-3-8B-Instruct rather than from the streaming-dialogue checkpoint.

This is not a loading bug. Before concluding anything we verified the adapter applies
correctly:

- active connector weights (`connector.modules_to_save.default.0.weight`) match the
  checkpoint file **exactly**, max abs diff `0.0`
- `lora_B` tensors are nonzero, so the adapter is genuinely in the forward path
- the `Some weights ... newly initialized: ['connector.*']` warning at load is benign —
  it refers to PEFT's unused `original_module` copy, not the active one

The failure mode is qualitative and consistent: the model emits Ego4D-style narration
instead of COIN taxonomy labels.

| Label | Model output |
|---|---|
| `install license plate frame` | " You were in a garage, and you fixed a car." |
| `assemble office chair` | " You hold a screwdriver." |
| `replace sim card` | " You look at the phone." |

The one "correct" answer in 15 samples was a Levenshtein accident: *"You put the laptop
on the table."* fuzzy-matched to `replace laptop screen` on the word "laptop".

---

## 2. Results so far

Zero-shot, released checkpoint, `coin_task_test`, on a 40-video subset:

| | 2 fps | 10 fps |
|---|---|---|
| Accuracy | **5.0%** (2/40) | **2.5%** (1/40) |
| Eval runtime | 68 s | 723 s |
| Mean tokens/video | 1,880 | 9,402 |
| Embeddings (40 videos) | 232 MB | 1.2 GB |

**The fps difference is noise, not signal.** 2/40 vs 1/40 is a one-video difference;
95% CIs are roughly [1.4%, 16.5%] and [0.4%, 12.9%]. The real result is that 10 fps
costs **10.6x the compute for no measurable gain** — the model fails on task format,
not temporal resolution.

### Two caveats when reporting any number from this pipeline

1. **Subset scores are optimistically biased.** `COINBenchmark.fuzzy_match` snaps every
   prediction to the nearest *surviving* category. The 40-video subset has 35 distinct
   task classes; the full test split has 180. Fewer distractors means easier matching,
   so full-split accuracy will likely be *lower*. To measure this, score against the
   full 180-class taxonomy rather than only classes present in the loaded annotations.
2. **~19% of COIN test videos are gone from YouTube** (measured: 40/50 and 29/36 batches
   succeeded). Report accuracy as "over N of 2797", not as a clean reproduction.
   `COIN.__init__` filters to videos found on disk, so missing videos silently shrink
   the eval set rather than erroring.

---

## 3. Open question worth resolving before any reproduction

**The evaluation prompt in the repo does not match the one in the paper.**

| Source | Prompt |
|---|---|
| Supplementary B.2 | `What task can summarize these steps?` + `[BenchEval]` suffix |
| `data/coin/benchmarks.py:113` | `What is the overall activity in the video? Format your answer concisely. No extra text output.` |

For a model fine-tuned and evaluated with the *same* prompt this is self-consistent and
harmless. But if you obtain the authors' COIN adapter and evaluate it with the repo's
prompt, you may query it with a prompt it never saw — producing a depressed number that
looks like a failed reproduction but is just a mismatch. **If in doubt, test both prompts
and take the higher.**

Note also that the paper's scoring is *stricter* than the repo's: it counts outputs
absent from the taxonomy as wrong, whereas `fuzzy_match` always snaps to something.

---

## 4. Upstream breakages fixed on this branch

The COIN path assumes 8 GPUs via submitit/torchrun and does not run as-shipped on current
dependencies.

| Problem | Fix |
|---|---|
| `download_videos.py` passes yt-dlp's removed `--username/--password`; also shuffles, so subsets aren't reproducible | new `data/coin/download_coin.py` |
| `distributed_ffmpeg` writes `videos_<fps>fps_max<res>`, but encode and `COIN` expect `videos_<fps>fps_<res>` | new `data/coin/preprocess_coin.py` |
| `distributed_encode` calls `torchvision.io.read_video`, removed in torchvision 0.29 | read frames via PyAV |
| `parse_args` validated pass 1 against base `LiveTrainingArguments`, rejecting subclass-only args (`--embed_mark`, `--max_num_frames`) | `models/__init__.py` — pass 1 now tolerates unknown args; pass 2 still strict |

**The directory name is load-bearing and fails silently.** `COIN.embed_dir` is computed as
`f"{video_root}_{embed_mark}_{vision_pretrained.replace('/','--')}"`, so embeddings must
land at exactly:

```
datasets/coin/videos_2fps_384_1+3x3_google--siglip-large-patch16-384/
```

Get it wrong and nothing errors — `COIN.__init__` finds zero videos and you score an
empty set.

---

## 5. Environment

| | |
|---|---|
| Machine | GB10 (DGX Spark), 119 GiB unified memory, single GPU, sm_121 |
| Env | `/home/s4ai/miniconda3/envs/videollm` — torch 2.14+cu130, transformers 4.55.4, peft 0.20, av, Levenshtein |
| Attention | **`sdpa`** — flash-attn is not installed and the README documents sdpa as the fallback |
| Models cached | `chenjoya/videollm-online-8b-v1plus` (adapter), `meta-llama/Meta-Llama-3-8B-Instruct` (base) |

The adapter is 1.8 GB / 899 M params / 455 tensors: ~353 M LoRA deltas, 525 M for the
resized `lm_head.base_layer` (the tokenizer adds a `<v>` vision placeholder), 21 M connector.
Base Llama-3 is frozen and downloaded separately — **the adapter is the entire difference
between 5% and 92.7%.**

---

## 6. Measured throughput

| Operation | Measured rate |
|---|---|
| yt-dlp download, 12 workers | 36 videos / 23 s |
| ffmpeg → 2fps/384 | ~141x realtime |
| SigLIP encode | 63.6 frames/s |
| Eval, 2 fps | ~1.7 s/sample |
| YouTube availability | ~80% |

### Dataset scale

| | Test | Train |
|---|---|---|
| Videos | 2,797 | 9,030 |
| Duration | 110 h | 358 h |
| Embeddings @2fps | ~16 GB | ~53 GB |
| Samples/epoch (all 5 benchmarks) | — | 123,784 |

Mean annotated span is 94 s (max 629 s), so `max_num_frames=1200` at 2 fps truncates
exactly **1 of 2797** test videos. At 10 fps that same cap covers only 120 s and would
truncate **796 videos (28.5%)** — hence `live1+_evaluate_1gpu.sh` defaults
`max_num_frames` to `fps*600`, holding the window at 600 s of video at any frame rate.

---

## 7. Estimated training cost

5 epochs on all five COIN benchmarks ≈ 285 M tokens ≈ **1.4 x 10^19 FLOPs**
(6N/token: forward, activation-gradients through frozen weights, checkpoint recompute).

| Hardware | Estimate |
|---|---|
| 8x A100 (paper's setup) | ~4–5 h |
| 1x A100 | ~30–35 h |
| **1x B200** | **~6–9 h** |
| 1x GB10 (here) | ~4 days |

**These are FLOP arithmetic with assumed MFU, not measurements — treat as ±2x.** Measure
real tokens/sec on a few dozen steps before committing to a long run. On a B200's 192 GB
you can raise `per_device_train_batch_size` well above 1 and cut grad-accum proportionally,
which should beat the estimate.

---

## 8. Current state

**Done:** test-split pipeline validated end to end on 40 videos, at both 2 and 10 fps.

**Not done:**
- Full test split — 40 of 2,797 videos downloaded
- Training split — nothing (29 videos exist only in a scratch dir)
- **No single-GPU training script yet.** `scripts/coin/live1+.sh` still assumes 8 GPUs +
  DeepSpeed. Needs: drop DeepSpeed (ZeRO-1 shards optimizer state across GPUs; pointless
  on one device), `gradient_accumulation_steps` 8 → 64 to preserve the effective batch of
  64, and `--attn_implementation sdpa`.
- `train.py` **has never been run on this stack.** Getting evaluation working took four
  fixes; expect similar breakages in `is_training=True` dataset construction,
  `get_learn_ranges`, and LoRA + gradient checkpointing under peft 0.20. De-risk with a
  short run on a handful of videos before committing to a long encode.

**In flight:** an email to the authors requesting the COIN LoRA adapter, with the prompt
and hyperparameter questions as fallback. If they share it, the ~50 h train-split pipeline
disappears and only the ~6 h test-split preprocessing remains.

---

## 9. Commands

### Setup (once)

```bash
# COIN ships annotations only; videos come from YouTube
mkdir -p datasets/coin
curl -sSL -o datasets/coin/coin.json \
  https://raw.githubusercontent.com/coin-dataset/annotations/master/COIN.json
```

All commands run from the repo root with the `videollm` env active.

### Test split — needed regardless of how the adapter is obtained

```bash
python -m data.coin.download_coin --split testing --num_workers 12   # ~0.5 h
python -m data.coin.preprocess_coin --stage all                      # ~4 h
bash scripts/coin/live1+_evaluate_1gpu.sh coin_task_test             # ~1 h
```

### Training split — only if reproducing the fine-tune

```bash
python -m data.coin.download_coin --split training --num_workers 12 \
  --manifest datasets/coin/download_manifest_train.json              # ~1.6 h
python -m data.coin.preprocess_coin --stage all                      # ~14 h
```

Both stages skip existing outputs, so runs are resumable and `--limit N` subsets are
strict prefixes of the full run (ids are sorted, not shuffled).

### Useful variations

```bash
# all five benchmarks
bash scripts/coin/live1+_evaluate_1gpu.sh \
  coin_step_test coin_next_test coin_task_test coin_procedure_test coin_taskprocedure_test

# 10 fps (requires: preprocess_coin.py --frame_fps 10)
FPS=10 bash scripts/coin/live1+_evaluate_1gpu.sh coin_task_test

# evaluate an adapter obtained from the authors
#   add --resume_from_checkpoint /path/to/adapter_dir to the torchrun line
```

### Sanity check for any COIN-finetuned adapter

Run step recognition alongside task summarization. If the adapter is genuinely the
Table 2a model, **Step should land near 63.1% and Task near 92.7%**. Both matching means
embeddings, prompts, tokenizer and scoring are all sound. One matching and not the other
points at a specific bug rather than a vague shortfall.

---

## 10. Git

Fork `S4AI-CornellTech/videollm-online`, `upstream` tracks `showlab/videollm-online`.

On this shared machine SSH offers `id_rsa` (Jared's) before `id_ed25519` (Xuesi's), so
GitHub silently authenticates as `JaredFern`, who lacks write access to the org. Fixed
per-repo, without touching the shared `~/.ssh/config` or `~/.gitconfig`:

```bash
git config --local core.sshCommand "ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes"
```

That is local config and does **not** travel with a clone — set it again elsewhere if needed.

`/datasets`, `/outputs`, `/checkpoints` are gitignored, so ~165 GB of video and embeddings
must move out of band (rsync ~69 GB of embeddings, or ~30 GB of 2fps/384 mp4s and re-encode
on the target).

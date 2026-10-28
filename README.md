# TransformerFromScratch

A from-scratch PyTorch implementation of the original encoder–decoder **Transformer** ("Attention Is All You Need") trained for **English → Hindi** translation. No prebuilt transformer layers — attention, positional encoding, layer norm, residual blocks, and the training loop are all implemented directly.

> **Status / notice:** the reference run is **20 epochs on a 50k-pair subset** (~6.9 h on a single T4). Full validation — BLEU/chrF over the whole validation split — was **not** run because of compute limits; validation is limited to 2 sample translations per epoch. This is a learning/from-scratch build, not a tuned translation system: **results would improve substantially with the full 1.66M pairs, more epochs, and a BPE tokenizer.** Raw log: [`logs/train_50k_t4.log`](logs/train_50k_t4.log).

---

## Architecture

| Component | Implementation |
| --- | --- |
| Embeddings | Learned, scaled by `sqrt(d_model)` |
| Positional encoding | Fixed sinusoidal (registered buffer) |
| Attention | Multi-head scaled dot-product, `d_model / h` per head |
| Blocks | Pre-norm residual blocks (post-norm in the paper) |
| Encoder / Decoder | `N` layers each, decoder adds cross-attention |
| Feed-forward | `Linear(d_model → d_ff) → ReLU → Dropout → Linear(d_ff → d_model)` |
| Output | Linear projection to target vocab (raw logits, no softmax) |

`config.py` currently runs the paper-sized model: `d_model=512`, `N=6`, `h=8`, `d_ff=512`, `dropout=0.01` — about 71M parameters with a 30k vocab (the projection and both embeddings dominate).

The model exposes `encode`, `decode`, and `project` separately so the encoder output can be reused during inference and attention can be inspected.

## Model size

For `d_model=512`, `N=6`, `d_ff=512`, `h=8` and a 30k word-level vocab: **71,357,744 parameters (~71.4M)**.

```
SELF-ATTENTION (per block)                     1,050,624
  Wq   512×512 + 512 bias                        262,656
  Wk                                             262,656
  Wv                                             262,656
  Wo                                             262,656

FEED-FORWARD (per block)                         525,312
  W1   512×d_ff + d_ff                           262,656
  W2   d_ff×512 + 512                            262,656

LAYER NORM (per block)                             2,048
  one LN = γ 512 + β 512 = 1,024 ; two per block  = 2 × 1,024

Encoder block = attn + FFN + 2 LN              1,577,984
Decoder block = self-attn + cross-attn + FFN + 3 LN
              = 1,050,624 + 1,050,624 + 525,312 + 3,072 = 2,629,632

6 × encoder block                              9,467,904
6 × decoder block                             15,777,792
source embedding 30000×512                    15,360,000
target embedding 30000×512                    15,360,000
projection 512×30000 + 30000                  15,390,000
final encoder + decoder LN                         2,048
                                              ----------
TOTAL                                         71,357,744
```

| Part | Params | Share |
|---|---|---|
| 6 encoder blocks | 9.47 M | 13.3% |
| 6 decoder blocks | 15.78 M | 22.1% |
| Embeddings (src + tgt) | 30.72 M | 43.0% |
| Output projection | 15.39 M | 21.6% |
| Final layer norms | 0.002 M | ~0% |

Notes:
- Attention and FFN together are only ~35% of the model; **65% is vocabulary** (embeddings + projection). That is why `vocab_size` dominates both parameter count and per-step cost.
- The positional encoding is a fixed buffer, not a parameter (0 params).
- At a 16k vocab the same layers give **49.8 M** (embeddings 16.38 M + projection 8.21 M).
- FP32 weights = `params × 4 B` → 0.27 GB at 30k, 0.19 GB at 16k; training state (weights + grads + Adam `m`/`v`) = ×3 → 0.80 GB / 0.56 GB.

## Project structure

```
config.py            Hyperparameters + checkpoint path helper
model.py             Transformer building blocks and build_transformer()
dataset.py           BilingualDataset, padding, causal mask
train.py             Data loading, tokenizers, training loop, greedy decode, validation
tokenizer_en.json    Trained WordLevel tokenizer (English)
tokenizer_hi.json    Trained WordLevel tokenizer (Hindi)
logs/                Reference training log (50k subset, 20 epochs)
requirements.txt     Dependencies
.env                 HF_TOKEN (git-ignored)
weights/             Per-epoch checkpoints (git-ignored)
runs/                TensorBoard logs, one timestamped subfolder per run (git-ignored)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` in the project root (optional, only needed for gated/rate-limited Hub access):

```
HF_TOKEN=hf_...
```

## Run

```bash
python train.py
```

TensorBoard:

```bash
tensorboard --logdir runs
```

If your terminal says `tensorboard` is not recognized or cannot find the command, run it through the Python environment where the project dependencies are installed:

```powershell
py -m tensorboard.main --logdir runs
```

On systems where `py` is unavailable, use `python -m tensorboard.main --logdir runs` instead. If you use a virtual environment, activate it first. Then open http://localhost:6006/ in your browser. TensorFlow is not required to view this project's PyTorch logs.

## Configuration

Edit `config.py`:

| Key | Default | Notes |
| --- | --- | --- |
| `batch_size` | `8` | T4 profile; use `2`–`4` on smaller GPUs |
| `epochs` | `20` | **~20.7 min/epoch measured** (50k pairs, AMP, T4) → ~6.9 h for 20 |
| `lr` | `1e-3` | Only the starting value; the Noam schedule sets the real LR per step |
| `warmup_steps` | `4000` | Steps to warm the LR up before decay |
| `seq_len` | `350` | Both source and target; longer sentences raise `ValueError` |
| `d_model` | `512` | Paper size; `128` trains much faster on a subset |
| `lang_src` / `lang_tgt` | `en` / `hi` | |
| `model_folder` | `weights` | |
| `model_basename` | `t_model` | Checkpoints: `weights/t_model00.pt`, … |
| `preload` | `None` | Epoch string to resume from, e.g. `"08"` (weights + optimizer restored) |
| `tokenizer_file` | `tokenizer_{0}.json` | `{0}` → language code |
| `experiment_name` | `runs/model` | TensorBoard log directory |
| `rebuild_tokenizer` | `False` | `True` once to retrain the tokenizers on the current data |
| `vocab_size` | `16000` | Vocab cap, only applied when the tokenizer is (re)built |
| `train_samples` | `50_000` | Pairs kept from the train split; `None` for all 1.66M |
| `repetition_penalty` | `2.0` | Validation-only nudge against token loops; `0.0` to disable |
| `use_amp` | `True` | fp16 autocast + GradScaler (T4 tensor cores); `False` for fp32 debugging |
| `seed` | `42` | Fixed seed for the shuffled subset and the 90/10 split |

## Data pipeline

- Dataset: [`cfilt/iitb-english-hindi`](https://huggingface.co/datasets/cfilt/iitb-english-hindi) — the IIT Bombay English–Hindi parallel corpus, `train` split (1,659,083 pairs), shuffled with a fixed seed and then capped at `train_samples`. Licence: **CC BY-NC 4.0** (non-commercial). It replaced `Helsinki-NLP/opus-100`, whose en–hi split is mostly subtitles and GNOME UI strings; the schema is identical (`translation` dict with `en`/`hi`), so `BilingualDataset` is unchanged.
- Split 90/10 into train/validation.
- Tokenizers use `WordLevel` + `Whitespace` pre-tokenizer, `min_frequency=2`, `vocab_size` cap, and special tokens `[UNK] [PAD] [SOS] [EOS]`. Cached JSONs are reused unless `rebuild_tokenizer` is enabled.
- Each sample is padded to `seq_len`:
  - encoder input: `[SOS] tokens [EOS] [PAD]…`
  - decoder input: `[SOS] tokens [PAD]…`
  - label: `tokens [EOS] [PAD]…` (shifted by one)
- Masks: padding mask for the encoder; padding + causal mask for the decoder.

## Training

- Loss: `CrossEntropyLoss(ignore_index=[PAD], label_smoothing=0.1)` on flattened logits.
- Optimizer: Adam (`eps=1e-9`, `foreach=False`) wrapped in a Noam warmup + inverse-sqrt schedule (`d_model^-0.5 · min(step^-0.5, step · warmup^-1.5)`).
- Mixed precision: fp16 autocast + `GradScaler` on CUDA (`use_amp`). The T4 has fp16 tensor cores but no bf16/TF32, so this is the main speedup.
- Training is single-process / single GPU. No `DataParallel` or DDP.
- Validation runs every epoch on a batch-size-1 loader, greedily decoding 2 samples and printing source / target / prediction.
- After every epoch the model, optimizer state, epoch, and global step are saved to `weights/t_modelXX.pt`. Resuming restores all four.

## Resource requirements

Estimated for the current config (`d_model=512`, `N=6`, `batch_size=8`, `seq_len=350`, AMP on, fp32 master weights):

| Item | 16k vocab | 30k vocab |
|---|---|---|
| Weights (fp32) | 0.19 GB | 0.27 GB |
| Gradients + Adam `m`/`v` | 0.37 GB | 0.53 GB |
| Activations (~200 MB/sample × 8) | ~1.6 GB | ~1.6 GB |
| Logits + cross-entropy | 0.17 GB | 0.31 GB |
| cuBLAS/cuDNN workspace + CUDA context | ~0.6 GB | ~0.6 GB |
| **Total incl. fragmentation** | **~3.5–4.5 GB** | **~4–5.5 GB** |

Fits a 15 GB Colab T4 with ~3× headroom; `batch_size: 16` lands around 7–8 GB. Formula, so you can rescale:

```
weights = grads = params × 4 B        Adam = 2 × params × 4 B
per layer ≈ 4·seq·d_model·4 B (Q/K/V/LN/FF) + 2·h·seq²·4 B (scores + softmax)
activations ≈ batch × (encoder + decoder layers) × per-layer cost
logits ≈ batch × seq × vocab_size × 4 B
```

Steps per epoch = `train_samples × 0.9 / batch_size`:
- 50k subset → 45,000 / 8 = **5,625 steps**
- full 1.66M → 1,493,174 / 8 = **186,647 steps** (~33× the 50k run)

Training time per epoch (50k subset, 45k samples):

| Config | h/epoch | 20 epochs |
|---|---|---|
| AMP, seq 350, 30k vocab — **measured** | **0.34** (20.7 min, 4.54 it/s) | **~6.9 h** |
| fp32, seq 350, 30k vocab (estimated) | 0.7–1.2 | 14–24 h |
| AMP, seq 150, 16k vocab (estimated) | 0.2–0.3 | 4–6 h |

Measured on 1×T4, Colab: 5,625 steps/epoch at 4.54 it/s. Add ~2–5 min for the dataset download/cache and the max-length scan.

VRAM does not depend on `train_samples` — the dataset lives in RAM, not VRAM (50k pairs ≈ a few hundred MB).

## Inference

`train.py` provides `greedy_decode`, which encodes the source once, then feeds the decoder one token at a time starting from `[SOS]`, appending the argmax token until `[EOS]` or `max_len`. An optional `repetition_penalty` subtracts a constant from already-generated tokens so validation output stays readable while the model is undertrained.

**Beam search is not implemented** — only greedy. Use beam search (`beam=4–5`) for a final demo or BLEU measurement, where ~1–2 extra BLEU is worth the k× decode time. Don't use it for per-epoch validation: it multiplies decoding cost by the beam width and tells you nothing new about training progress.

## Visualization & metrics (next steps)

Nothing below is implemented — this is the "can be added" list, roughly in cost order:

- **Loss / LR curves** — already logged. `tensorboard --logdir runs` shows `train loss` and `lr` (warmup + decay). First thing to look at.
- **Validation loss / perplexity** — currently only 2 sample decodes per epoch. Averaging `loss_fn` over the validation loader is a ~10-line addition and far more informative than eyeballing 2 sentences.
- **BLEU / chrF** — `sacrebleu` over the whole validation split. Expensive (needs full decoding), so run it once at the end, not per epoch.
- **Attention maps** — every `MultiHeadAttentionBlock` already keeps `self.attention_scores` from the last forward pass. Plot encoder self-attention (source↔source) and decoder cross-attention (target↔source) as heatmaps with `matplotlib` to see whether alignment is learned.
- **Gradient norm** — cheap health check; log `clip_grad_norm_`'s return value and plot it next to the loss.
- **Sample grid in TensorBoard** — write a few source/target/prediction triples with `writer.add_text` so runs stay comparable.

## Known limitations

- Trained on **50k of 1.66M pairs for 20 epochs**; the loss is still noisy (3.86–5.53) and sentences are often unrelated to the source. More data and epochs are the main lever.
- **No BLEU/chrF.** Validation is 2 decoded samples per epoch, so there is no held-out quality number.
- Checkpoints saved before the LayerNorm change have `(1,)`-shaped `gamma`/`beta` and will not load into the current model.
- `seq_len` pads every sample to the full length, so short sentences still cost `seq_len` steps.

## License

MIT — see `LICENSE`.

The training data ([`cfilt/iitb-english-hindi`](https://huggingface.co/datasets/cfilt/iitb-english-hindi)) is **CC BY-NC 4.0** — non-commercial use only, and it requires attribution to Kunchukuttan, Mehta & Bhattacharyya, *The IIT Bombay English-Hindi Parallel Corpus*, LREC 2018. The MIT licence above covers this code only, not the data or any weights trained on it.

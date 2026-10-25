# TransformerFromScratch

A from-scratch PyTorch implementation of the original encoder–decoder **Transformer** ("Attention Is All You Need") trained for **English → Hindi** translation. No prebuilt transformer layers — attention, positional encoding, layer norm, residual blocks, and the training loop are all implemented directly.

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

## Project structure

```
config.py            Hyperparameters + checkpoint path helper
model.py             Transformer building blocks and build_transformer()
dataset.py           BilingualDataset, padding, causal mask
train.py             Data loading, tokenizers, training loop, greedy decode, validation
tokenizer_en.json    Trained WordLevel tokenizer (English)
tokenizer_hi.json    Trained WordLevel tokenizer (Hindi)
requirements.txt     Dependencies
.env                 HF_TOKEN (git-ignored)
weights/             Per-epoch checkpoints (git-ignored)
runs/                TensorBoard logs (git-ignored)
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
| `epochs` | `20` | ~1–1.5 h/epoch at 100k pairs on a T4 |
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
| `train_samples` | `100_000` | Pairs kept from the train split; `None` for all |
| `repetition_penalty` | `2.0` | Validation-only nudge against token loops; `0.0` to disable |
| `use_amp` | `True` | fp16 autocast + GradScaler (T4 tensor cores); `False` for fp32 debugging |
| `seed` | `42` | Fixed seed for the shuffled subset and the 90/10 split |

## Data pipeline

- Dataset: `Helsinki-NLP/opus-100`, config `en-hi`, `train` split, shuffled with a fixed seed and then capped at `train_samples`.
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

Steps per epoch = `train_samples × 0.9 / batch_size`, e.g. 90,000 / 8 = **11,250 steps** for the 100k subset, or 60,500 for the full 538k.

## Inference

`train.py` provides `greedy_decode`, which encodes the source once, then feeds the decoder one token at a time starting from `[SOS]`, appending the argmax token until `[EOS]` or `max_len`. An optional `repetition_penalty` subtracts a constant from already-generated tokens so validation output stays readable while the model is undertrained.

**Beam search is not implemented** — only greedy. Use beam search (`beam=4–5`) for a final demo or BLEU measurement, where ~1–2 extra BLEU is worth the k× decode time. Don't use it for per-epoch validation: it multiplies decoding cost by the beam width and tells you nothing new about training progress.

## Known limitations

- No BLEU or other metric is computed; quality is judged from the printed samples.
- Checkpoints saved before the LayerNorm change have `(1,)`-shaped `gamma`/`beta` and will not load into the current model.
- `seq_len` pads every sample to the full length, so short sentences still cost `seq_len` steps.

## License

MIT — see `LICENSE`.

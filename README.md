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

Defaults in `build_transformer`: `d_model=128`, `N=3`, `h=4`, `d_ff=512`, `dropout=0.01`.

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

## Configuration

Edit `config.py`:

| Key | Default | Notes |
| --- | --- | --- |
| `batch_size` | `2` | Small to fit limited GPU memory |
| `epochs` | `20` | |
| `lr` | `1e-3` | Adam, `eps=1e-9`, `foreach=False` |
| `seq_len` | `350` | Both source and target; longer sentences raise `ValueError` |
| `d_model` | `128` | Paper uses `512` |
| `lang_src` / `lang_tgt` | `en` / `hi` | |
| `model_folder` | `weights` | |
| `model_basename` | `t_model` | Checkpoints: `weights/t_model00.pt`, … |
| `preload` | `None` | Epoch string to resume from, e.g. `"08"` |
| `tokenizer_file` | `tokenizer_{0}.json` | `{0}` → language code |
| `experiment_name` | `runs/model` | TensorBoard log directory |

## Data pipeline

- Dataset: `Helsinki-NLP/opus-100`, config `en-hi`. The split is set to `"test"` for light local runs — switch to `"train"` for real training.
- Split 90/10 into train/validation.
- Tokenizers are built once with `WordLevel` + `Whitespace` pre-tokenizer, `min_frequency=2`, and special tokens `[UNK] [PAD] [SOS] [EOS]`. They are cached to JSON and reused on later runs.
- Each sample is padded to `seq_len`:
  - encoder input: `[SOS] tokens [EOS] [PAD]…`
  - decoder input: `[SOS] tokens [PAD]…`
  - label: `tokens [EOS] [PAD]…` (shifted by one)
- Masks: padding mask for the encoder; padding + causal mask for the decoder.

## Training

- Loss: `CrossEntropyLoss(ignore_index=[PAD], label_smoothing=0.1)` on flattened logits.
- Optimizer: Adam with `lr=1e-3`.
- Validation runs every epoch on a batch-size-1 loader, greedily decoding 2 samples and printing source / target / prediction.
- After every epoch the model, optimizer state, epoch, and global step are saved to `weights/t_modelXX.pt`.

## Inference

`train.py` provides `greedy_decode`, which encodes the source once, then feeds the decoder one token at a time starting from `[SOS]`, appending the argmax token until `[EOS]` or `max_len`.

## Known limitations

- `preload` restores epoch, optimizer, and global step, but does **not** call `model.load_state_dict`, so weights are not actually restored. Add that call before relying on resume.
- Only a checkpoint/step system is implemented; no BLEU or other metric is computed.
- Evaluation reuses the `test` split by default, so reported outputs are not a held-out benchmark.

## License

MIT — see `LICENSE`.

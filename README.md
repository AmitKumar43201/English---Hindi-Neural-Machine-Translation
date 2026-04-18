# English → Hindi Neural Machine Translation

A sequence-to-sequence neural machine translation system built using a **Transformer architecture implemented from scratch** in PyTorch — no pre-built transformer libraries. The model is trained on an English-Hindi parallel corpus and translates English sentences into Hindi.

---

## Architecture

The model follows the original Transformer architecture from [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), with an encoder-decoder design.

```
Input (English)
      │
      ▼
┌─────────────────────┐
│  SentencePiece BPE  │  ← Tokenization
│  Token Embeddings   │
│  Positional Encoding│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    Encoder Stack    │  ← 2 layers
│  ┌───────────────┐  │
│  │ Multi-Head    │  │  ← 8 heads
│  │ Self-Attention│  │
│  ├───────────────┤  │
│  │  Feed-Forward │  │  ← d_ff = 2048
│  │  Network      │  │
│  └───────────────┘  │
└────────┬────────────┘
         │  encoder output
         ▼
┌─────────────────────┐
│    Decoder Stack    │  ← 2 layers
│  ┌───────────────┐  │
│  │ Masked Multi- │  │  ← causal mask
│  │ Head Attention│  │
│  ├───────────────┤  │
│  │  Cross-       │  │  ← attends to encoder output
│  │  Attention    │  │
│  ├───────────────┤  │
│  │  Feed-Forward │  │
│  │  Network      │  │
│  └───────────────┘  │
└────────┬────────────┘
         │
         ▼
  Linear + Softmax
         │
         ▼
  Output (Hindi)
```

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Total Parameters | ~27M |
| Embedding Dimension (d_model) | 512 |
| Attention Heads | 8 |
| Encoder Layers | 2 |
| Decoder Layers | 2 |
| Feed-Forward Dimension | 2048 |
| Max Sequence Length | 32 |
| Dropout | 0.1 |

---

## Project Structure

```
├── utils.py          # Positional encoding, embeddings, attention, masking
├── encoder.py        # Encoder layer and Encoder stack
├── decoder.py        # Decoder layer and Decoder stack
├── transformer.py    # Full Transformer model
├── data_loader.py    # Dataset, DataLoader, train/val split
└── train.py          # Training loop with validation
```

---

## Dataset

- **Source:** English-Hindi parallel sentence pairs
- **Size:** 74,500 sentence pairs
- **Split:** 95% train (~70,800) / 5% validation (~3,700)
- **Tokenization:** SentencePiece BPE — separate models trained for English and Hindi

---

## Training Details

| Setting | Value |
|---------|-------|
| Optimizer | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |
| Learning Rate | Warmup scheduler (4,000 warmup steps) |
| Loss Function | CrossEntropyLoss with PAD-token masking |
| Batch Size | 16 |
| Epochs | 10 |
| Gradient Clipping | max norm = 1.0 |
| Decoding | Greedy decoding |
| Hardware | Google Colab (T4 GPU) |

### Learning Rate Schedule

The learning rate follows the warmup schedule from the original paper:

```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
```

This linearly increases the learning rate for the first `warmup_steps` steps, then decays it proportionally to the inverse square root of the step number.

### Key Training Techniques

- **Teacher Forcing** — During training, the decoder receives ground truth Hindi tokens as input rather than its own predictions, enabling faster and more stable training.
- **Causal (Look-ahead) Masking** — Prevents decoder positions from attending to future positions during self-attention, ensuring autoregressive generation.
- **Padding Masks** — Applied to both encoder and decoder to prevent attention over PAD tokens.
- **Model Checkpointing** — Model, optimizer, and scheduler states are saved after every epoch, allowing training to be resumed.

---

## Results

| Metric | Value |
|--------|-------|
| Train Loss | ~X.XX |
| Validation Loss | ~X.XX |

*Replace with actual values after training completes.*

---

## Sample Translations

| English | Hindi (Predicted) |
|---------|-------------------|
| what are you doing? | आप क्या कर रहे हैं? |
| i don't belong here! | मैं ... |

*Results improve with more epochs and cleaner training data.*

---

## Setup

### Requirements

```bash
pip install torch sentencepiece
```

### Train SentencePiece Tokenizers

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='train.en',
    model_prefix='en_tokenizer',
    vocab_size=8000
)

spm.SentencePieceTrainer.train(
    input='train.hi',
    model_prefix='hi_tokenizer',
    vocab_size=8000
)
```

### Train the Model

```python
# Set paths and hyperparameters in train.py, then run:
python train.py
```

---

## Tech Stack

- **Python** · **PyTorch** · **SentencePiece** · **NumPy**
- Trained on **Google Colab (T4 GPU)**

---

## References

- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS.

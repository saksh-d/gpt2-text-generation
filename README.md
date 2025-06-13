# Mini-GPT Text Generation

A from-scratch implementation of a decoder-only Transformer for character-level text generation in PyTorch.

---

## Project Overview

This project is a learning-focused, end-to-end implementation of a GPT-style Transformer model trained from scratch. It includes:

- A bigram language model for name generation (baseline)
- A mini GPT-style Transformer trained on Shakespeare
- Clean, minimal PyTorch implementations
- Generated outputs that mimic Shakespearean dialogue

---
## Models Included

### 1. Bigram Language Model
- Trained on a list of names (`data/names.txt`)
- Uses simple character-to-character transition probabilities
- Generates plausible fictional names
- Serves as a baseline before moving to neural models

Run:
```bash
python bigram.py
```

### 2. GPT-style Transformer (char-level)
Trained from scratch on `data/input.txt` (Tiny Shakespeare)

Implements:
- Multi-head self-attention (with mask)
- Causal masking
- LayerNorm
- Feedforward layer
- Training loop with validation split

Outputs high-quality Shakespearean dialogue

Run:
```bash
python mini-gpt.py
```

---
## Results

After ~5000 steps of training:
- Training loss: 0.859
- Validation loss: ~1.57
- Output shows structured verse, punctuation, speaker labels

View `results.png`

# Mini LLM — Character-Level Transformer in PyTorch

A lightweight, decoder-only Transformer (GPT-style autoregressive language model) built from scratch in PyTorch. Designed to train quickly on a standard laptop CPU/GPU while demonstrating core LLM mechanics: token/positional embeddings, causal multi-head self-attention, residual connections, and temperature-scaled text generation.

---

## Features
- **From-Scratch Architecture:** Custom implementation of scaled dot-product causal self-attention, multi-head attention, pre-layer normalization (`LayerNorm`), and GELU feed-forward blocks without high-level Transformer wrappers.
- **Laptop-Friendly Training:** Tuned with compact dimensions so training loss convergence and text generation can be observed in minutes on consumer hardware.
- **Autoregressive Generation:** Supports temperature-scaled multinomial sampling (`temperature`) to control creativity vs. predictability.
- **Model Checkpointing:** Saves trained weights (`mini_llm.pt`) for reuse and inference.

---

## Project Structure
```text
mini-llm/
├── data/
│   └── input.txt       # Raw UTF-8 training corpus (ignored by git)
├── model.py            # Transformer architecture (Attention, Blocks, MiniLLM)
├── train.py            # Vocabulary builder, training loop, evaluation & generation
├── .gitignore          # Excludes virtual environment, checkpoints, and raw data
└── README.md           # Project documentation
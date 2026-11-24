# Mini LLM — GPT-Style Transformer in PyTorch

A lightweight, decoder-only Transformer (GPT-style autoregressive language model) built from scratch in PyTorch. Trains on a standard laptop CPU/GPU and demonstrates core LLM mechanics: token/positional embeddings, causal multi-head self-attention, GPT-2 BPE subword tokenization (`tiktoken`), checkpoint serialization, and temperature / top-k / top-p (nucleus) text generation.


## Features
- **From-Scratch Transformer Architecture:** Custom scaled dot-product causal self-attention, multi-head attention, pre-layer normalization (`LayerNorm`), and GELU feed-forward blocks without high-level Transformer wrappers.
- **GPT-2 Subword Tokenization:** Uses `tiktoken` (`gpt2` BPE encoding, `vocab_size = 50,257`) for subword modeling.
- **Self-Contained Checkpoints:** Saves model weights, architecture hyperparameters, and tokenizer metadata in `checkpoint.pt`.
- **Standalone CLI Inference (`generate.py`):** Supports custom prompts, token length, temperature scaling, top-k filtering, and top-p (nucleus) sampling.


## Project Structure
```text
mini-llm/
├── data/
│   └── input.txt       # Raw UTF-8 training corpus (ignored by git)
├── model.py            # Transformer architecture & sampling (Attention, Blocks, MiniLLM)
├── train.py            # BPE encoding, training loop, validation & checkpoint saving
├── generate.py         # CLI inference script with temperature, top-k, and top-p controls
├── requirements.txt    # Python dependencies (torch, numpy, tiktoken)
├── LICENSE             # MIT License
├── .gitignore          # Excludes virtual environment, checkpoints, and raw data
└── README.md           # Project documentation

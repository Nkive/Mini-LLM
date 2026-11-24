import argparse
import tiktoken
import torch
from model import MiniLLM


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained MiniLLM checkpoint.")
    parser.add_argument("--prompt", type=str, default="To be, or not", help="Starting text prompt")
    parser.add_argument("--tokens", type=int, default=120, help="Number of new subword tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (higher = more random)")
    parser.add_argument("--top-k", type=int, default=50, help="Keep top-k tokens (0 to disable)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling cumulative probability threshold")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="Path to saved model checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    enc = tiktoken.get_encoding(checkpoint.get("tokenizer", "gpt2"))

    model = MiniLLM(
        vocab_size=checkpoint["vocab_size"],
        n_embd=checkpoint.get("n_embd", 128),
        num_heads=checkpoint.get("num_heads", 4),
        n_layer=checkpoint.get("n_layer", 4),
        block_size=checkpoint["block_size"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    encoded = enc.encode(args.prompt) or [enc.eot_token]
    context = torch.tensor([encoded], dtype=torch.long, device=device)
    output_ids = model.generate(
        context,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p,
    )[0].tolist()

    print("\n--- Generated Text ---")
    print(enc.decode(output_ids))


if __name__ == "__main__":
    main()
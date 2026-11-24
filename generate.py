import argparse
import torch
from model import MiniLLM


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained MiniLLM checkpoint.")
    parser.add_argument("--prompt", type=str, default="ROMEO:", help="Starting text prompt")
    parser.add_argument("--tokens", type=int, default=300, help="Number of new characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (higher = more random)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="Path to saved model checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    block_size = checkpoint["block_size"]

    model = MiniLLM(
        vocab_size=checkpoint["vocab_size"],
        block_size=block_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    encoded = [stoi[c] for c in args.prompt if c in stoi]
    if not encoded:
        encoded = [0]

    context = torch.tensor([encoded], dtype=torch.long, device=device)
    output_ids = model.generate(context, max_new_tokens=args.tokens, temperature=args.temperature)[0].tolist()
    generated_text = "".join(itos[i] for i in output_ids)

    print("\n--- Generated Text ---")
    print(generated_text)


if __name__ == "__main__":
    main()
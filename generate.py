import argparse
import tiktoken
import torch
from model import MiniLLM


def generate_response(model, enc, prompt, max_tokens, temperature, top_k, top_p, device):
    encoded = enc.encode(prompt) or [enc.eot_token]
    context = torch.tensor([encoded], dtype=torch.long, device=device)
    output_ids = model.generate(
        context,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
        top_p=top_p,
    )[0].tolist()
    return enc.decode(output_ids)


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained MiniLLM checkpoint.")
    parser.add_argument("--prompt", type=str, default="To be, or not", help="Starting text prompt")
    parser.add_argument("--tokens", type=int, default=120, help="Number of new subword tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (higher = more random)")
    parser.add_argument("--top-k", type=int, default=40, help="Keep top-k tokens (0 to disable)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling cumulative probability threshold")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="Path to saved model checkpoint")
    parser.add_argument("--interactive", action="store_true", help="Run an interactive prompt loop in the terminal")
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

    if args.interactive:
        print("Interactive mode ready. Type a prompt and press Enter (type 'exit' or 'quit' to stop).\n")
        while True:
            try:
                user_prompt = input("Prompt > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

            if not user_prompt:
                continue
            if user_prompt.lower() in {"exit", "quit"}:
                break

            result = generate_response(
                model, enc, user_prompt, args.tokens, args.temperature, args.top_k, args.top_p, device
            )
            print(f"\n{result}\n" + "-" * 40)
    else:
        result = generate_response(
            model, enc, args.prompt, args.tokens, args.temperature, args.top_k, args.top_p, device
        )
        print("\n--- Generated Text ---")
        print(result)


if __name__ == "__main__":
    main()
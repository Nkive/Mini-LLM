import torch
from model import MiniLLM

# Hyperparameters
batch_size = 32
block_size = 64
max_iters = 1500
eval_interval = 300
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]


def get_batch(split):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, eval_iters=50):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


model = MiniLLM(vocab_size=vocab_size, block_size=block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

checkpoint = {
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size,
    "block_size": block_size,
    "stoi": stoi,
    "itos": itos,
}
torch.save(checkpoint, "checkpoint.pt")
print("Saved checkpoint to checkpoint.pt")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n--- Generated Sample ---")
print(decode(model.generate(context, max_new_tokens=300, temperature=0.8)[0].tolist()))
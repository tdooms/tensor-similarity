"""Generate stories from a trained AttentionLM checkpoint."""

import argparse
import yaml
import torch
from lib import AttentionLM, get_tokenizer


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str = "",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cuda",
) -> str:
    """Autoregressive generation with top-k sampling."""
    model.eval()

    if prompt:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    else:
        # Start with a random token
        input_ids = torch.randint(0, model.vocab_size, (1, 1), device=device)

    n_ctx = model.n_ctx

    for _ in range(max_new_tokens):
        # Truncate to context window
        x = input_ids[:, -n_ctx:]

        logits = model(x)
        next_logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            topk_vals, _ = torch.topk(next_logits, top_k)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            next_logits[next_logits < threshold] = float("-inf")

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Stop on EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate stories from AttentionLM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--n-stories", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    model = AttentionLM.from_config(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint (step {ckpt['step']})\n")

    for i in range(args.n_stories):
        story = generate(
            model, tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print(f"--- Story {i+1} ---")
        print(story)
        print()


if __name__ == "__main__":
    main()

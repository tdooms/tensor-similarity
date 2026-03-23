"""Train bilinear MLP models on MNIST across multiple seeds, saving checkpoints.

Outputs to artifacts/convergence/:
  - checkpoints_{seed}.pt: list of {"batch": int, "state_dict": dict}
  - history_{seed}.pt: {"batch": [], "train_loss": [], "train_acc": [], "val_acc": []}

Runtime: ~2 min per seed, ~10 min total.
"""
import torch
from copy import deepcopy
from torchvision import datasets, transforms

from src.paper.shared import ARTIFACT_DIR, make_model

OUT = ARTIFACT_DIR / "convergence"

SEEDS = [1, 2, 3, 42, 99]
EPOCHS = 20
BATCH_SIZE = 256
CHECKPOINT_EVERY = 10  # batches


def train_one_seed(seed, train_loader, test_loader):
    model = make_model(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    checkpoints = []
    history = {"batch": [], "train_loss": [], "train_acc": [], "val_acc": []}
    batch_idx = 0

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_idx += 1

            if batch_idx % CHECKPOINT_EVERY == 0:
                checkpoints.append({"batch": batch_idx, "state_dict": deepcopy(model.state_dict())})

                acc = (logits.argmax(-1) == y).float().mean().item()
                model.eval()
                with torch.no_grad():
                    val_correct = sum((model(vx).argmax(-1) == vy).sum().item() for vx, vy in test_loader)
                    val_total = sum(len(vy) for _, vy in test_loader)
                model.train()

                history["batch"].append(batch_idx)
                history["train_loss"].append(loss.item())
                history["train_acc"].append(acc)
                history["val_acc"].append(val_correct / val_total)

        print(f"  seed={seed} epoch={epoch+1}/{EPOCHS} "
              f"loss={history['train_loss'][-1]:.4f} val_acc={history['val_acc'][-1]:.4f}")

    return checkpoints, history


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST("_data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("_data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024, shuffle=False)

    for seed in SEEDS:
        print(f"\n=== Training seed {seed} ===")
        checkpoints, history = train_one_seed(seed, train_loader, test_loader)
        torch.save(checkpoints, OUT / f"checkpoints_{seed}.pt")
        torch.save(history, OUT / f"history_{seed}.pt")
        print(f"  Saved {len(checkpoints)} checkpoints")


if __name__ == "__main__":
    main()

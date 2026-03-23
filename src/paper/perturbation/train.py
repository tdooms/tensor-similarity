"""Train with progressive digit addition/removal curriculum.

Curriculum: train on {0-4}, add 5, ..., add 9, remove 9, re-add 9.
Each stage continues from the previous stage's final weights.

Outputs to artifacts/perturbation/:
  - checkpoints_{stage_name}.pt: list of {"batch": int, "state_dict": dict}
  - history_{stage_name}.pt: training metrics per stage
  - curriculum.pt: the curriculum definition

Runtime: ~2 min per stage × 8 stages ≈ 15 min total.
"""
import torch
from copy import deepcopy
from torchvision import datasets, transforms

from src.paper.shared import ARTIFACT_DIR, make_model

OUT = ARTIFACT_DIR / "perturbation"

CURRICULUM = [
    {"name": "base",     "digits": list(range(5))},
    {"name": "add_5",    "digits": list(range(6))},
    {"name": "add_6",    "digits": list(range(7))},
    {"name": "add_7",    "digits": list(range(8))},
    {"name": "add_8",    "digits": list(range(9))},
    {"name": "add_9",    "digits": list(range(10))},
    {"name": "remove_9", "digits": list(range(9))},
    {"name": "readd_9",  "digits": list(range(10))},
]

SEED = 42
EPOCHS_PER_STAGE = 15
BATCH_SIZE = 256
CHECKPOINT_EVERY = 10


def make_loader(digits, train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data = datasets.MNIST("_data", train=train, download=True, transform=transform)
    mask = torch.tensor([y in digits for y in data.targets])
    data.data = data.data[mask]
    data.targets = data.targets[mask]
    return torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE if train else 1024, shuffle=train)


def train_stage(model, optimizer, train_loader, test_loader, stage_name):
    loss_fn = torch.nn.CrossEntropyLoss()
    checkpoints = []
    history = {"batch": [], "train_loss": [], "train_acc": [], "val_acc": []}
    batch_idx = 0

    for epoch in range(EPOCHS_PER_STAGE):
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
                    vc = sum((model(vx).argmax(-1) == vy).sum().item() for vx, vy in test_loader)
                    vt = sum(len(vy) for _, vy in test_loader)
                model.train()

                history["batch"].append(batch_idx)
                history["train_loss"].append(loss.item())
                history["train_acc"].append(acc)
                history["val_acc"].append(vc / vt)

        print(f"  {stage_name} epoch={epoch+1}/{EPOCHS_PER_STAGE} "
              f"val_acc={history['val_acc'][-1]:.4f}")

    return checkpoints, history


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    model = make_model(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for stage in CURRICULUM:
        name, digits = stage["name"], stage["digits"]
        print(f"\n=== Stage: {name} (digits {digits}) ===")

        train_loader = make_loader(digits, train=True)
        test_loader = make_loader(digits, train=False)
        checkpoints, history = train_stage(model, optimizer, train_loader, test_loader, name)

        torch.save(checkpoints, OUT / f"checkpoints_{name}.pt")
        torch.save(history, OUT / f"history_{name}.pt")
        print(f"  Saved {len(checkpoints)} checkpoints")

    torch.save(CURRICULUM, OUT / "curriculum.pt")


if __name__ == "__main__":
    main()

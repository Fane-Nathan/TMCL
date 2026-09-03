from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import SyntheticSequenceDataset
from .model import CarBrain, ModelConfig


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = ModelConfig(**cfg["model"])
    model = CarBrain(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    syn = cfg["synthetic"]
    dataset = SyntheticSequenceDataset(
        size=max(cfg["train_steps"] * cfg["batch_size"], 10_000),
        sequence_length=cfg["sequence_length"],
        dt=syn["dt"],
        steering_gain_range=tuple(syn["train_steering_gain"]),
        action_noise_std_range=tuple(syn["train_action_noise_std"]),
        seed=cfg["seed"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    model.train()
    for step, (tokens, actions) in enumerate(loader, start=1):
        tokens = tokens.to(device)
        actions = actions.to(device)
        pred = model(tokens)
        loss = F.mse_loss(pred, actions)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 100 == 0:
            print(f"step={step:05d} loss={loss.item():.6f}")
        if step >= cfg["train_steps"]:
            break

    out_dir = Path(cfg["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    out_path = out_dir / "car_brain.safetensors"
    save_file(weights, str(out_path))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import deque

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from .model import CarBrain, ModelConfig
from .synthetic import Dynamics, SyntheticLaneEnv


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rollout(
    model: CarBrain,
    *,
    dynamics: Dynamics,
    horizon: int,
    dt: float,
    context_len: int,
    use_history: bool,
    seed: int,
    device: torch.device,
) -> float:
    env = SyntheticLaneEnv(dynamics=dynamics, dt=dt, seed=seed)
    state = env.reset()
    prev_action = 0.0
    prev_reward = 0.0
    prev_done = 0.0
    history: deque[np.ndarray] = deque(maxlen=context_len)
    total = 0.0

    for _ in range(horizon):
        token = np.concatenate(
            [state, np.asarray([prev_action, prev_reward, prev_done], dtype=np.float32)]
        ).astype(np.float32)
        if use_history:
            history.append(token)
        else:
            history.clear()
            history.append(token)

        tokens = torch.from_numpy(np.stack(history))[None].to(device)
        with torch.no_grad():
            action = float(model.act(tokens).cpu().numpy()[0, 0])

        state, reward, done, _ = env.step(action)
        total += reward
        prev_action = action
        prev_reward = reward
        prev_done = float(done)
        if done:
            break

    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/car_brain.safetensors")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CarBrain(ModelConfig(**cfg["model"])).to(device)
    model.load_state_dict(load_file(args.checkpoint, device=str(device)))
    model.eval()

    syn = cfg["synthetic"]
    for latency in syn["test_latency_steps"]:
        full, no_hist = [], []
        for i in range(args.episodes):
            dynamics = Dynamics(steering_gain=1.0, action_noise_std=0.0, latency_steps=int(latency))
            seed = cfg["seed"] + 10_000 * int(latency) + i
            full.append(
                rollout(
                    model,
                    dynamics=dynamics,
                    horizon=syn["horizon"],
                    dt=syn["dt"],
                    context_len=cfg["sequence_length"],
                    use_history=True,
                    seed=seed,
                    device=device,
                )
            )
            no_hist.append(
                rollout(
                    model,
                    dynamics=dynamics,
                    horizon=syn["horizon"],
                    dt=syn["dt"],
                    context_len=cfg["sequence_length"],
                    use_history=False,
                    seed=seed,
                    device=device,
                )
            )

        full_m = float(np.mean(full))
        no_m = float(np.mean(no_hist))
        gap = full_m - no_m
        print(
            f"latency={latency} full_history={full_m:.3f} "
            f"no_history={no_m:.3f} adaptation_gap={gap:+.3f}"
        )


if __name__ == "__main__":
    main()

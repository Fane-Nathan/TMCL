from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Dynamics:
    steering_gain: float = 1.0
    action_noise_std: float = 0.0
    latency_steps: int = 0


class SyntheticLaneEnv:
    """Tiny lane-following environment with unobserved vehicle dynamics.

    State = [lateral_error, heading_error, speed, curvature].
    Action = steering in [-1, 1].
    Hidden dynamics modify the executed steering command.
    """

    def __init__(self, dynamics: Dynamics, dt: float = 0.05, seed: int | None = None):
        self.dynamics = dynamics
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self._queue: deque[float] = deque()
        self.reset()

    def reset(self) -> np.ndarray:
        self.y = float(self.rng.normal(0.0, 0.15))
        self.heading = float(self.rng.normal(0.0, 0.05))
        self.speed = float(self.rng.uniform(0.7, 1.0))
        self.phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.t = 0
        self._queue = deque([0.0] * self.dynamics.latency_steps)
        return self._obs()

    def _curvature(self) -> float:
        return 0.45 * np.sin(self.phase + self.t * self.dt * 0.8)

    def _obs(self) -> np.ndarray:
        return np.asarray([self.y, self.heading, self.speed, self._curvature()], dtype=np.float32)

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        commanded = float(np.clip(action, -1.0, 1.0))
        if self.dynamics.latency_steps > 0:
            self._queue.append(commanded)
            executed = self._queue.popleft()
        else:
            executed = commanded

        executed *= self.dynamics.steering_gain
        if self.dynamics.action_noise_std > 0:
            executed += float(self.rng.normal(0.0, self.dynamics.action_noise_std))

        curvature = self._curvature()
        self.heading += self.dt * (1.8 * executed - 1.2 * curvature - 0.35 * self.heading)
        self.y += self.dt * self.speed * np.sin(self.heading)
        self.speed = float(np.clip(self.speed + self.dt * (0.05 - 0.03 * abs(executed)), 0.5, 1.1))
        self.t += 1

        reward = 1.0 - 1.8 * abs(self.y) - 0.35 * abs(self.heading) - 0.02 * abs(commanded)
        done = abs(self.y) > 2.0
        return self._obs(), float(reward), done, {"executed_action": executed}


def expert_action(state: np.ndarray, dynamics: Dynamics) -> float:
    """Privileged controller used only to synthesize offline demonstrations."""
    y, heading, _speed, curvature = [float(x) for x in state]
    desired = 0.95 * curvature - 1.4 * y - 0.9 * heading
    gain = max(abs(dynamics.steering_gain), 0.15)
    return float(np.clip(desired / gain, -1.0, 1.0))

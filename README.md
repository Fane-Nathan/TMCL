# TMCL — TrackMania Context Learning

TMCL is an experimental research codebase for testing whether a pretrained driving sequence model can adapt **in context** to hidden vehicle dynamics without gradient updates, then transfer that representation into TMRL/TrackMania.

The immediate goal is deliberately smaller than a full TrackMania foundation model:

1. Generate many offline driving episodes with hidden dynamics changes.
2. Pretrain a causal sequence model on those episodes.
3. Save the resulting "car brain" as a tensor checkpoint (`.safetensors`).
4. Freeze the checkpoint at evaluation time.
5. Compare the **same weights** with full history versus history reset every step.
6. Only after the mechanism works, connect the pretrained encoder to TMRL.

## Core hypothesis

A policy trained across sufficiently diverse episode-level dynamics can learn to infer the active dynamics from its own action-consequence history:

```text
state_t, previous_action, previous_reward
                  |
                  v
          causal Transformer
                  |
        hidden context / belief
                  |
                  v
              action_t
```

At test time the parameters stay fixed. Adaptation is therefore expressed through context, not SGD:

`theta(t=0) == theta(t=T)`, while the hidden/context state changes with experience.

The primary ICL metric is the adaptation gap:

`G = return(full_history) - return(no_history)`

A positive gap on a held-out fault mechanism is evidence that history itself is useful, rather than the policy merely being robust.

## Phase 0: synthetic proof of concept

The first benchmark is a cheap synthetic lane-following task. Each episode has unobserved vehicle dynamics such as steering gain, steering bias/noise, or action latency. The model observes only driving state and its interaction history.

Training and evaluation use disjoint dynamics distributions. In particular, action latency can be held out as a cross-mechanism test.

This phase answers one question before we spend time collecting TrackMania data:

> Can the architecture learn an in-context driving adaptation mechanism at all?

## Phase 1: offline driving pretraining

Once Phase 0 works, add trajectory adapters for public/offline driving datasets and TrackMania replay/ghost telemetry. The canonical episode representation should eventually support:

- image or visual embedding
- speed / gear / RPM
- previous action
- previous reward
- action
- termination flag
- optional privileged training metadata (map, surface, velocity, position, hidden fault)

Privileged metadata should not be exposed to the policy during held-out ICL evaluation.

## Phase 2: TMRL integration

Load the pretrained checkpoint into the TMRL sequence agent:

```text
car_brain.safetensors
        |
        v
vision/timestep encoder + causal context model
        |
   initially frozen
        |
        v
REDQ-SAC actor/critic heads
```

Recommended ablations:

| Variant | Pretrained | Context encoder | History |
|---|---:|---|---:|
| Scratch | No | Train | Yes |
| Frozen | Yes | Frozen | Yes |
| Slow FT | Yes | Low LR | Yes |
| No-history | Yes | Same weights | No |
| Full FT | Yes | Full LR | Yes |

The most important comparison is **Slow FT vs No-history** under held-out tracks or dynamics.

## Planned repository layout

```text
TMCL/
├── configs/
│   └── synthetic.yaml
├── src/tmcl/
│   ├── data.py
│   ├── model.py
│   ├── synthetic.py
│   ├── train.py
│   └── evaluate.py
├── tests/
├── pyproject.toml
└── README.md
```

## Research principles

- Keep train/test fault mechanisms explicitly disjoint when testing cross-mechanism ICL.
- Evaluate identical weights with and without history.
- Do not call ordinary domain robustness "in-context learning."
- Save normalization statistics with every checkpoint.
- Prefer small falsifiable experiments before expensive TrackMania collection.
- Track random seeds and report per-seed results.

## Status

Initial research scaffold. Phase 0 synthetic benchmark is being implemented first.

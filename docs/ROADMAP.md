# Quadruped RL engineering roadmap

## Current checkpoint

- Existing quadruped models, simulation environments, analytical trot baseline,
  and historical outputs remain available.
- Shared PPO now uses differentiable PyTorch losses and Adam updates.
- NumPy networks remain inference-only and are rejected by the PPO trainer.
- The isolated repair passes 64 tests and a bounded contextual-bandit learning gate.
- Earlier PPO/DR learning and compute-budget conclusions are withdrawn because
  the historical shared backend did not update weights.

## Required next milestones

1. Run and inspect the PPO repair CI on the published commit.
2. Audit production collectors, termination/truncation handling, reset behavior,
   reward definitions, and evaluation metrics before fresh locomotion training.
3. Demonstrate locomotion on nominal physics with valid training provenance.
4. Evaluate DR with matched training budgets and a fixed evaluation protocol.
5. Validate deployment interfaces, timing, and sensor assumptions separately.

The shared buffer accepts explicit final-observation bootstrap values on truncation.
This repair does not certify every legacy collector's use of that interface.
Vectorized trainers, residual control, and heading experiments are outside this PR.
No physical hardware transfer or real-time deployment is established here.
Historical coverage percentages are not current repair coverage measurements.

See [PPO repair scope and reproduction](PPO_REPAIR.md).

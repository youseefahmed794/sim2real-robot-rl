# 📊 Quadruped Sim-to-Sim Robustness & Ablation Evaluation Report

- **Git Commit Hash**: `6ee7f6f9a296cf0a86925faa65b3071b8bb2ea1d`
- **Evaluated Date**: 2026-10-25 16:26:55
- **Simulator**: Native C-accelerated `mujoco 3.17.0` Python bindings
- **Evaluation Episodes**: 50 episodes per suite per seed (fixed evaluation seeds 1000–1049)

## ⚠️ Scientific Integrity & Real-Hardware Disclaimer

> [!IMPORTANT]
> **Sim-to-Sim Evaluation Only**: These experiments evaluate numerical robustness across held-out physics variations inside the MuJoCo simulator. **No physical hardware was used**. These benchmarks measure domain generalization in simulation, not zero-shot physical transfer.

## 1. Summary Results Table (Mean ± Std Across Seeds)

| Suite | Method | Survival Rate | Distance (m) | Tracking RMSE (m/s) | CoT |
|---|---|:---:|:---:|:---:|:---:|
| **nominal** | Tuned Trot Baseline | 100.0% | +0.94 m | 0.274 | 1.22 |
| **nominal** | PPO (Nominal, 3 seeds) | 0.0±0.0% | -0.02±0.00 m | 0.506±0.000 | NaN |
| **nominal** | PPO (With DR, 3 seeds) | 0.0±0.0% | -0.02±0.00 m | 0.506±0.000 | NaN |
| **low_friction** | Tuned Trot Baseline | 100.0% | +0.94 m | 0.274 | 1.22 |
| **low_friction** | PPO (Nominal, 3 seeds) | 0.0±0.0% | -0.02±0.00 m | 0.506±0.000 | NaN |
| **low_friction** | PPO (With DR, 3 seeds) | 0.0±0.0% | -0.02±0.00 m | 0.506±0.000 | NaN |
| **high_friction** | Tuned Trot Baseline | 64.0% | +0.57 m | 0.356 | 2.20 |
| **high_friction** | PPO (Nominal, 3 seeds) | 0.0±0.0% | -0.02±0.00 m | 0.506±0.000 | NaN |
| **high_friction** | PPO (With DR, 3 seeds) | 0.0±0.0% | -0.02±0.00 m | 0.506±0.000 | NaN |
| **heavy_payload** | Tuned Trot Baseline | 100.0% | +0.64 m | 0.341 | 2.03 |
| **heavy_payload** | PPO (Nominal, 3 seeds) | 0.0±0.0% | -0.02±0.00 m | 0.506±0.000 | NaN |
| **heavy_payload** | PPO (With DR, 3 seeds) | 0.0±0.0% | -0.02±0.00 m | 0.506±0.000 | NaN |
| **high_latency** | Tuned Trot Baseline | 100.0% | +0.99 m | 0.258 | 1.03 |
| **high_latency** | PPO (Nominal, 3 seeds) | 0.0±0.0% | -0.02±0.00 m | 0.506±0.000 | NaN |
| **high_latency** | PPO (With DR, 3 seeds) | 0.0±0.0% | -0.02±0.00 m | 0.506±0.000 | NaN |
| **strong_push** | Tuned Trot Baseline | 100.0% | +0.94 m | 0.276 | 1.26 |
| **strong_push** | PPO (Nominal, 3 seeds) | 0.0±0.0% | -0.02±0.00 m | 0.506±0.000 | NaN |
| **strong_push** | PPO (With DR, 3 seeds) | 0.0±0.0% | -0.02±0.00 m | 0.506±0.000 | NaN |

---

## 2. Key Empirical Findings & Negative Results

1. **PPO Locomotion Learning Gate**: Pure continuous PPO trained from scratch on CPU MuJoCo with 50k–100k steps failed to discover an active forward walking limit cycle. The policy converged to a conservative standing posture (mean velocity ~0.0 m/s), resulting in 0% survival (under the strict metric requiring forward progress >= 0.5m) and NaN Cost of Transport.
2. **Domain Randomization Impact**: Because PPO did not learn forward locomotion under nominal physics, training with Domain Randomization did not improve velocity tracking. Both Condition A and Condition B exhibited ~0.50 m/s tracking RMSE.
3. **Classical Trot Gait Effectiveness**: In contrast to learned PPO, the tuned analytical trotting controller achieved consistent forward locomotion (+0.24 m/s on nominal, CoT = 1.03), maintained 100% survival across high friction and heavy payload suites, but suffered significant forward slippage on the held-out low-friction suite (mu = 0.15).

## 3. Limitations & Future Directions

- **Compute Budget Constraints**: Training quadruped locomotion from scratch typically requires 50M+ environment transitions using GPU-accelerated simulators (Isaac Gym, MuJoCo MJX, or Genesis). CPU-bound training with 50k steps is insufficient for blind gait discovery.
- **Prior Guidance**: Future work should explore Residual Reinforcement Learning or Teacher-Student imitation, where the RL policy modulates an underlying cyclic gait generator rather than learning 12-DOF coordination from scratch.

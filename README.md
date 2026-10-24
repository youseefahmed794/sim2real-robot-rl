# 🦿 Sim-to-Real RL for Dynamic Quadruped Locomotion (PPO + Domain Randomization)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy_Jalisco-3498db?logo=ros)](https://docs.ros.org/en/jazzy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/Status-Day%201%20Active%20Sprint-blue.svg)](docs/ROADMAP.md)

A production-grade, research-oriented **Sim-to-Real Reinforcement Learning framework** for dynamic legged robot locomotion. Designed to train agile, robust locomotion policies on a 12-DOF quadruped robot and transfer zero-shot to physical systems via comprehensive **Domain Randomization**, observation delay compensation, and a real-time ROS 2 deployment pipeline.

Developed by **[Youseef Abdel Tawwab](https://github.com/youseefahmed794)** as part of the [Robotics & Embodied AI Graduate Portfolio Roadmap](https://github.com/youseefahmed794).

---

## 📑 Table of Contents

- [Motivation & Core Insights](#-motivation--core-insights)
- [System Architecture](#-system-architecture)
- [Progressive 7-Day Roadmap](#-progressive-7-day-roadmap)
- [Theoretical & Mathematical Formulation](#-theoretical--mathematical-formulation)
  - [1. Markov Decision Process (MDP) for Legged Locomotion](#1-markov-decision-process-mdp-for-legged-locomotion)
  - [2. Proximal Policy Optimization (PPO) with GAE](#2-proximal-policy-optimization-ppo-with-gae)
  - [3. Sim-to-Real Domain Randomization Mechanics](#3-sim-to-real-domain-randomization-mechanics)
- [Robot Kinematic & Dynamic Specification](#-robot-kinematic--dynamic-specification)
- [Installation & Getting Started](#-installation--getting-started)
- [Scientific References](#-scientific-references)
- [License & Contact](#-license--contact)

---

## 🌟 Motivation & Core Insights

Deploying reinforcement learning policies trained in simulation directly to physical quadruped robots (*Sim-to-Real Transfer*) presents severe challenges:
1. **Unmodeled Actuator Dynamics**: High-speed brushless DC motors exhibit non-linear friction, thermal derating, and rotor inertia discrepancies.
2. **Contact Non-Smoothness**: Ground reaction forces and terrain coefficient of restitution vary unpredictably.
3. **Sensor Latency & Noise**: Microcontroller SPI/CAN communication loops introduce 10–30 ms of stochastic delay.

This project addresses these fundamental gaps through:
- **Massive Dynamics Randomization**: Joint damping, link mass ($\pm 15\%$), ground friction ($\mu \in [0.2, 1.2]$), and push disturbances.
- **Latency Emulation Buffer**: Training with randomized observation history queues to build latency-invariant policies.
- **Comparative Baseline**: Direct benchmarking against a classical Model Predictive Control (Convex MPC) framework.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Sim["Simulation Environment (MuJoCo / Physics)"]
        PHYS["Rigid Body Dynamics (RNEA / ABA)"]
        CONTACT["Ground Reaction Contact Solver"]
        RAND["Domain Randomization Engine"]
        RAND --> PHYS
    end

    subgraph Agent["Deep RL Agent (PyTorch)"]
        ACTOR["Actor Network: pi_theta(a | s)"]
        CRITIC["Critic Network: V_phi(s)"]
        PPO["PPO-Clip Loss + GAE Advantage"]
    end

    subgraph Deploy["ROS 2 Real-Time Deployment"]
        NODE["ROS 2 Inference Node (50 Hz)"]
        JS["/joint_states & /imu/data"]
        CMD["/joint_group_controller/commands"]
    end

    PHYS -->|Observations s_t| ACTOR
    ACTOR -->|Joint Targets q_des| PHYS
    PHYS --> CONTACT
    PPO --> ACTOR
    PPO --> CRITIC
    ACTOR -.->|Exported TorchScript/ONNX| NODE
    JS --> NODE --> CMD
```

---

## 📅 Progressive 7-Day Roadmap

Development is structured into daily verified milestones:

| Day | Focus Area | Status | Deliverables |
|---|---|---|---|
| **Day 1** | **Robot Modeling & Kinematics** | 🚀 **Completed** | 12-DOF URDF + MJCF models, $SO(3)$ spatial math, unit tests |
| **Day 2** | **Gymnasium Locomotion Env** | ⏳ In Progress | State/action spaces, multi-component reward formulation |
| **Day 3** | **PPO Policy & Rollout Buffer** | ⏳ Planned | Actor-Critic networks, vectorized GAE buffer, training loop |
| **Day 4** | **Sim-to-Real Randomization** | ⏳ Planned | Mass, friction, damping, push impulses, delay buffer |
| **Day 5** | **Classical MPC Baseline** | ⏳ Planned | Simplified MPC comparison, energy/stability benchmarks |
| **Day 6** | **Test Suite & CI Pipeline** | ⏳ Planned | 100% Pytest test coverage, GitHub Actions workflow |
| **Day 7** | **ROS 2 Real-Time Inference** | ⏳ Planned | ROS 2 Jazzy node, real-time loop, full documentation |

See [docs/ROADMAP.md](docs/ROADMAP.md) for detailed architectural specifications.

---

## 🔬 Theoretical & Mathematical Formulation

### 1. Markov Decision Process (MDP) for Legged Locomotion

We formulate locomotion as an infinite-horizon discounted Markov Decision Process $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:
- **Observation Space $\boldsymbol{s}_t \in \mathbb{R}^{48}$**:
  $$\boldsymbol{s}_t = \left[ \boldsymbol{\omega}_b, \boldsymbol{g}_b, \boldsymbol{v}_{\text{cmd}}, (\boldsymbol{q} - \boldsymbol{q}_{\text{nominal}}), \dot{\boldsymbol{q}}, \boldsymbol{a}_{t-1} \right]$$
  where $\boldsymbol{\omega}_b$ is base angular velocity, $\boldsymbol{g}_b = \boldsymbol{R}_b^T \boldsymbol{e}_z$ is projected gravity vector, $\boldsymbol{v}_{\text{cmd}} = [v_{x,\text{cmd}}, v_{y,\text{cmd}}, \omega_{z,\text{cmd}}]^T$ is user velocity command, $\boldsymbol{q} \in \mathbb{R}^{12}$ are joint positions, and $\dot{\boldsymbol{q}} \in \mathbb{R}^{12}$ are joint velocities.

- **Action Space $\boldsymbol{a}_t \in \mathbb{R}^{12}$**: Target joint angles passed to low-level PD controllers:
  $$\boldsymbol{\tau}_t = \boldsymbol{K}_p (\boldsymbol{q}_{\text{nominal}} + k_{\text{scale}} \boldsymbol{a}_t - \boldsymbol{q}) - \boldsymbol{K}_d \dot{\boldsymbol{q}}$$

- **Composite Reward Function $\mathcal{R}_t$**:
  $$\mathcal{R}_t = w_{\text{lin}} r_{\text{lin}} + w_{\text{ang}} r_{\text{ang}} + w_{\text{energy}} r_{\text{energy}} + w_{\text{smooth}} r_{\text{smooth}} + w_{\text{orient}} r_{\text{orient}}$$

---

## 🤖 Robot Kinematic & Dynamic Specification

- **Total Mass**: $12.5\,\text{kg}$ (Body: $6.0\,\text{kg}$, Legs: $4 \times 1.625\,\text{kg}$)
- **Degrees of Freedom**: 12 (3 actuated revolute joints per leg: Abduction/Adduction Hip, Hip Pitch, Knee Pitch)
- **Torque Limit**: $\pm 35.5\,\text{N}\cdot\text{m}$ per joint
- **Maximum Velocity**: $21.0\,\text{rad/s}$
- **Nominal Standing Height**: $0.28\,\text{m}$

---

## 📄 License & Contact

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**Author**: Youseef Abdel Tawwab  
**Email**: [youseefahmed794@gmail.com](mailto:youseefahmed794@gmail.com)  
**GitHub**: [@youseefahmed794](https://github.com/youseefahmed794)

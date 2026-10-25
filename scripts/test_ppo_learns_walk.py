"""
Verify that PPO on Nominal MuJoCo Physics learns forward locomotion:
- Mean forward velocity > 0.3 m/s at commanded 0.5 m/s
- Return curve rises over training steps
- Saves learning curve and renders rollout video to results/videos/ppo_nominal_walk.mp4

Author: Youseef Abdel Tawwab
"""

import os
import math
import time
import numpy as np
import mujoco
import cv2
import matplotlib.pyplot as plt

from sim2real_rl.envs.mujoco_env import MuJoCoQuadrupedEnv
from sim2real_rl.envs.rewards import RewardConfig, RewardManager
from sim2real_rl.algo.network import ActorCritic
from sim2real_rl.algo.buffer import RolloutBuffer
from sim2real_rl.algo.ppo import PPO, PPOConfig
from sim2real_rl.utils.metrics import (
    compute_mechanical_power,
    compute_cost_of_transport,
    evaluate_survival,
)
from sim2real_rl.utils.spatial_math import quat_to_euler


def create_locomotion_reward_config():
    """Tuned reward config ensuring forward velocity tracking dominates over standing still."""
    return RewardConfig(
        w_lin_vel=3.0,
        sigma_lin_vel=0.30,
        w_ang_vel=0.5,
        sigma_ang_vel=0.50,
        w_lin_vel_z=-1.0,
        w_ang_vel_xy=-0.05,
        w_orientation=-1.0,
        w_base_height=-5.0,
        nominal_height=0.325,  # Matches MuJoCo model equilibrium height
        w_torques=-1e-4,
        w_mechanical_power=-1e-4,
        w_action_rate=-0.01,
        w_joint_acc=-1e-7,
        w_joint_limits=-5.0,
        w_foot_slip=-0.02,
        w_alive=0.2,           # Small alive bonus so standing still does not dominate
    )


def train_ppo_nominal(
    total_steps: int = 100000,
    rollout_len: int = 500,
    seed: int = 42,
    lr: float = 3e-4,
):
    os.makedirs("results/videos", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    np.random.seed(seed)
    reward_cfg = create_locomotion_reward_config()
    env = MuJoCoQuadrupedEnv(
        reward_config=reward_cfg,
        max_episode_steps=rollout_len,
        kp=90.0,
        kd=3.0,
    )
    
    dt = env.model.opt.timestep * env.decimation
    print(f"Training PPO Nominal for {total_steps} steps (rollout_len={rollout_len}, dt={dt:.4f}s)...")
    
    ppo_cfg = PPOConfig(
        num_epochs=4,
        mini_batch_size=64,
        learning_rate=lr,
        clip_param=0.2,
        entropy_coef=0.01,
    )
    agent = PPO(config=ppo_cfg, observation_dim=48, action_dim=12)
    buffer = RolloutBuffer(buffer_size=rollout_len, observation_dim=48, action_dim=12)

    returns_history = []
    vel_history = []
    collected_steps = 0
    t0 = time.time()

    while collected_steps < total_steps:
        cmd_vel = np.array([0.5, 0.0, 0.0], dtype=np.float64)
        obs, info = env.reset(seed=seed + collected_steps, cmd_vel=cmd_vel)
        ep_ret = 0.0
        ep_vels = []

        for step in range(rollout_len):
            action, log_prob, val = agent.actor_critic.act(obs, deterministic=False)
            next_obs, rew, term, trunc, step_info = env.step(action)
            done = term or trunc

            buffer.add(obs, action, rew, done, val, log_prob)
            ep_ret += rew
            ep_vels.append(step_info["forward_vel"])
            obs = next_obs
            collected_steps += 1

            if done or buffer.full:
                last_val = 0.0 if term else float(agent.actor_critic.get_value(obs))
                buffer.compute_returns_and_advantages(last_value=last_val)
                agent.train_step(buffer)
                buffer.clear()
                break

        returns_history.append(ep_ret)
        mean_v = float(np.mean(ep_vels)) if ep_vels else 0.0
        vel_history.append(mean_v)

        if len(returns_history) % 20 == 0:
            avg_ret = np.mean(returns_history[-20:])
            avg_vel = np.mean(vel_history[-20:])
            elapsed = time.time() - t0
            print(f"  Step {collected_steps}/{total_steps} ({collected_steps/elapsed:.0f} fps) | Return (last 20): {avg_ret:.2f} | Forward Vel: {avg_vel:.3f} m/s")

    # Evaluate learned policy on 20s rollout
    print("\nEvaluating learned PPO policy on 20s rollout...")
    eval_env = MuJoCoQuadrupedEnv(
        reward_config=reward_cfg,
        max_episode_steps=int(math.ceil(20.0 / dt)),
        kp=90.0,
        kd=3.0,
    )
    renderer = mujoco.Renderer(eval_env.model, height=360, width=480)
    video_frames = []
    
    obs, info = eval_env.reset(seed=999, cmd_vel=np.array([0.5, 0.0, 0.0]))
    eval_vels = []
    eval_powers = []
    start_x = eval_env.data.qpos[0]
    has_torso_contact = False

    floor_id = mujoco.mj_name2id(eval_env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    base_geom_id = mujoco.mj_name2id(eval_env.model, mujoco.mjtObj.mjOBJ_GEOM, "base_geom")

    for s in range(eval_env.max_episode_steps):
        if s % 5 == 0:
            renderer.update_scene(eval_env.data)
            video_frames.append(renderer.render())

        act, _, _ = agent.actor_critic.act(obs, deterministic=True)
        obs, r, term, trunc, sinfo = eval_env.step(act)
        eval_vels.append(sinfo["forward_vel"])

        torques = eval_env.data.ctrl[:]
        joint_vels = eval_env.data.qvel[6:18]
        eval_powers.append(compute_mechanical_power(torques, joint_vels))

        for c in range(eval_env.data.ncon):
            con = eval_env.data.contact[c]
            if (con.geom1 == base_geom_id and con.geom2 == floor_id) or \
               (con.geom2 == base_geom_id and con.geom1 == floor_id):
                has_torso_contact = True

        if term:
            break

    final_x = eval_env.data.qpos[0]
    dist = final_x - start_x
    mean_eval_v = float(np.mean(eval_vels)) if eval_vels else 0.0
    mean_power = float(np.mean(eval_powers)) if eval_powers else 0.0
    total_mass = float(np.sum(eval_env.model.body_mass))
    cot = compute_cost_of_transport(mean_power, total_mass, 9.81, mean_eval_v)
    survived = evaluate_survival(has_torso_contact, dist, min_distance_threshold=0.5)

    # Save video
    video_path = "results/videos/ppo_nominal_walk.mp4"
    if video_frames:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, 25.0, (480, 360))
        for f in video_frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Saved PPO rollout video to {video_path}")

    # Save training curves plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(returns_history, label="Episode Return", color="blue")
    plt.title("PPO Nominal Return Curve")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(vel_history, label="Forward Velocity (m/s)", color="green")
    plt.axhline(y=0.3, color="red", linestyle="--", label="Target (0.3 m/s)")
    plt.title("PPO Forward Velocity")
    plt.xlabel("Episode")
    plt.ylabel("Mean Forward Vel (m/s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/ppo_nominal_learning_curve.png")
    plt.close()

    print("\n--- PPO NOMINAL EVALUATION SUMMARY ---")
    print(f"Total distance traveled: {dist:.4f} m")
    print(f"Mean forward velocity: {mean_eval_v:.4f} m/s (Target: > 0.30 m/s)")
    print(f"Tracking error: {abs(mean_eval_v - 0.5):.4f} m/s")
    print(f"Cost of Transport (CoT): {cot}")
    print(f"Torso contact: {has_torso_contact} | Survived: {survived}")

    return {
        "mean_forward_velocity": mean_eval_v,
        "distance": dist,
        "cot": cot,
        "survived": survived,
        "agent": agent,
    }


if __name__ == "__main__":
    train_ppo_nominal(total_steps=100000)

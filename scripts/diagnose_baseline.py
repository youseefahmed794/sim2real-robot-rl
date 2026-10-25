"""
Diagnose the Classical Baseline on Nominal MuJoCo Physics.

Runs a 20 s rollout at command vx = 0.5 m/s, logs:
- actual mean forward velocity
- base height (mean, min, max)
- total distance traveled (final x - start x)
- mean mechanical power: mean of sum_i max(tau_i * qdot_i, 0)
- CoT (computed with strictly defined formula)
- torso ground contact check
- survival determination
- renders video to results/videos/baseline_rollout.mp4 (or .gif)

Author: Youseef Abdel Tawwab
"""

import os
import math
import numpy as np
import mujoco
import cv2

from sim2real_rl.envs.mujoco_env import MuJoCoQuadrupedEnv
from sim2real_rl.baselines.trot_controller import AnalyticalTrotController
from sim2real_rl.utils.metrics import (
    compute_mechanical_power,
    compute_cost_of_transport,
    evaluate_survival,
)
from sim2real_rl.utils.spatial_math import quat_to_euler


def diagnose_baseline(
    stride_freq: float = 1.5,
    step_height: float = 0.08,
    nominal_height: float = 0.325,
    kp: float = 90.0,
    kd: float = 3.0,
    video_name: str = "tuned_baseline_rollout.mp4",
):
    os.makedirs("results/videos", exist_ok=True)
    
    # 20 seconds rollout
    env = MuJoCoQuadrupedEnv(kp=kp, kd=kd)
    dt = env.model.opt.timestep * env.decimation
    target_time = 20.0  # seconds
    total_steps = int(math.ceil(target_time / dt))
    
    env.max_episode_steps = total_steps
    controller = AnalyticalTrotController(
        stride_frequency=stride_freq,
        step_height=step_height,
        nominal_height=nominal_height,
    )

    cmd_vel = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    obs, info = env.reset(seed=42, cmd_vel=cmd_vel)
    controller.reset()

    total_mass = float(np.sum(env.model.body_mass))
    g = 9.81
    start_x = float(env.data.qpos[0])

    forward_vels = []
    base_heights = []
    pitch_angles = []
    roll_angles = []
    mechanical_powers = []
    tracking_errors = []
    has_torso_contact = False

    base_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "base_geom")
    floor_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # MuJoCo renderer for visualization
    renderer = mujoco.Renderer(env.model, height=360, width=480)
    video_frames = []

    print(f"=== Diagnosing Classical Trot Baseline for {target_time}s ({total_steps} steps, dt={dt:.4f}s) ===")
    print(f"Params: f={stride_freq}Hz, h={step_height}m, d={nominal_height}m, kp={kp}, kd={kd}")
    print(f"Commanded vx: {cmd_vel[0]} m/s | Total mass: {total_mass:.2f} kg")

    terminated_at_step = None

    for step in range(total_steps):
        # Render every 5th step (~25 fps)
        if step % 5 == 0:
            renderer.update_scene(env.data)
            frame = renderer.render()
            video_frames.append(frame)

        # Classical controller action
        base_pitch = info.get("base_pitch", 0.0)
        base_roll = info.get("base_roll", 0.0)
        action = controller.get_action(dt=dt, cmd_vel=cmd_vel, base_pitch=base_pitch, base_roll=base_roll)

        obs, reward, terminated, truncated, step_info = env.step(action)

        # Log state
        base_pos = env.data.qpos[0:3]
        base_quat = env.data.qpos[3:7]
        roll, pitch, _ = quat_to_euler(base_quat)
        
        forward_vel = float(step_info["forward_vel"])
        forward_vels.append(forward_vel)
        base_heights.append(float(base_pos[2]))
        pitch_angles.append(float(pitch))
        roll_angles.append(float(roll))
        tracking_errors.append(abs(forward_vel - cmd_vel[0]))

        # Mechanical power
        torques = env.data.ctrl[:]
        joint_vels = env.data.qvel[6:18]
        p_mech = compute_mechanical_power(torques, joint_vels)
        mechanical_powers.append(p_mech)

        # Check torso-ground contact
        for c in range(env.data.ncon):
            con = env.data.contact[c]
            if (con.geom1 == base_geom_id and con.geom2 == floor_geom_id) or \
               (con.geom2 == base_geom_id and con.geom1 == floor_geom_id):
                has_torso_contact = True

        if terminated and terminated_at_step is None:
            terminated_at_step = step
            print(f"Episode terminated early at step {step} ({step * dt:.2f}s). Reason: height={base_pos[2]:.3f}, roll={roll:.3f}, pitch={pitch:.3f}")
            break

    final_x = float(env.data.qpos[0])
    distance_traveled = final_x - start_x
    mean_forward_vel = float(np.mean(forward_vels)) if forward_vels else 0.0
    mean_mech_power = float(np.mean(mechanical_powers)) if mechanical_powers else 0.0
    mean_tracking_err = float(np.mean(tracking_errors)) if tracking_errors else 0.0
    cot = compute_cost_of_transport(
        mean_mechanical_power=mean_mech_power,
        total_mass=total_mass,
        gravity=g,
        mean_forward_velocity=mean_forward_vel,
    )

    # Minimum threshold distance for 20 s at 0.5 m/s (expected 10m): threshold 0.5 m
    survived = evaluate_survival(
        has_torso_contact=has_torso_contact,
        distance_traveled=distance_traveled,
        min_distance_threshold=0.5,
    )

    # Save video
    video_path = f"results/videos/{video_name}"
    if video_frames:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, 25.0, (480, 360))
        for f in video_frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Saved rollout video to {video_path} ({len(video_frames)} frames)")

    print("\n--- BASELINE DIAGNOSIS REPORT ---")
    print(f"Total time simulated: {len(forward_vels) * dt:.2f} s / {target_time:.2f} s")
    print(f"Start X: {start_x:.4f} m | Final X: {final_x:.4f} m")
    print(f"Total distance traveled: {distance_traveled:.4f} m")
    print(f"Actual mean forward velocity: {mean_forward_vel:.4f} m/s (Commanded: {cmd_vel[0]} m/s)")
    print(f"Velocity tracking error (RMSE): {mean_tracking_err:.4f} m/s")
    print(f"Base height: mean={np.mean(base_heights):.4f} m, min={np.min(base_heights):.4f} m, max={np.max(base_heights):.4f} m")
    print(f"Base pitch (deg): mean={np.degrees(np.mean(pitch_angles)):.2f}°, max={np.degrees(np.max(np.abs(pitch_angles))):.2f}°")
    print(f"Base roll (deg): mean={np.degrees(np.mean(roll_angles)):.2f}°, max={np.degrees(np.max(np.abs(roll_angles))):.2f}°")
    print(f"Mean mechanical power: {mean_mech_power:.4f} W")
    print(f"Cost of Transport (CoT): {cot}")
    print(f"Torso ground contact detected: {has_torso_contact}")
    print(f"Survived (no torso contact & distance >= 0.5m): {survived}")

    return {
        "distance_traveled": distance_traveled,
        "mean_forward_velocity": mean_forward_vel,
        "mean_tracking_error": mean_tracking_err,
        "mean_base_height": float(np.mean(base_heights)),
        "mean_mechanical_power": mean_mech_power,
        "cot": cot,
        "has_torso_contact": has_torso_contact,
        "survived": survived,
    }


if __name__ == "__main__":
    diagnose_baseline()

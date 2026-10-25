"""
Bounded Systematic Parameter Tuning for Classical Trot Gait Baseline on Nominal MuJoCo Physics.

Search budget: 36 parameter combinations across:
- Stride frequency: [1.5, 2.0, 2.5] Hz
- Swing step height: [0.06, 0.08, 0.10] m
- Stance extension depth: [0.325, 0.340] m
- Low-level PD gains: (kp=55, kd=1.8) vs (kp=90, kd=3.0)

Evaluates 5-second rollouts (to bound compute time) at commanded vx = 0.5 m/s.
Logs distance, mean velocity, tracking error, torso contact, and survival.

Author: Youseef Abdel Tawwab
"""

import itertools
import math
import numpy as np
import mujoco

from sim2real_rl.envs.mujoco_env import MuJoCoQuadrupedEnv
from sim2real_rl.baselines.trot_controller import AnalyticalTrotController
from sim2real_rl.utils.metrics import (
    compute_mechanical_power,
    compute_cost_of_transport,
    evaluate_survival,
)


def evaluate_trot_config(
    stride_freq: float,
    step_height: float,
    stance_depth: float,
    kp: float,
    kd: float,
    duration: float = 5.0,
    cmd_vx: float = 0.5,
) -> dict:
    env = MuJoCoQuadrupedEnv(kp=kp, kd=kd)
    dt = env.model.opt.timestep * env.decimation
    steps = int(math.ceil(duration / dt))
    env.max_episode_steps = steps

    ctrl = AnalyticalTrotController(
        stride_frequency=stride_freq,
        step_height=step_height,
        nominal_height=stance_depth,
    )
    cmd_vel = np.array([cmd_vx, 0.0, 0.0], dtype=np.float64)
    obs, info = env.reset(seed=42, cmd_vel=cmd_vel)
    ctrl.reset()

    start_x = float(env.data.qpos[0])
    forward_vels = []
    powers = []
    has_torso_contact = False

    base_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "base_geom")
    floor_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    for step in range(steps):
        action = ctrl.get_action(dt=dt, cmd_vel=cmd_vel)
        obs, reward, terminated, truncated, step_info = env.step(action)

        forward_vels.append(step_info["forward_vel"])
        
        torques = env.data.ctrl[:]
        joint_vels = env.data.qvel[6:18]
        powers.append(compute_mechanical_power(torques, joint_vels))

        for c in range(env.data.ncon):
            con = env.data.contact[c]
            if (con.geom1 == base_geom_id and con.geom2 == floor_geom_id) or \
               (con.geom2 == base_geom_id and con.geom1 == floor_geom_id):
                has_torso_contact = True

        if terminated:
            break

    final_x = float(env.data.qpos[0])
    dist = final_x - start_x
    mean_v = float(np.mean(forward_vels)) if forward_vels else 0.0
    mean_p = float(np.mean(powers)) if powers else 0.0
    cot = compute_cost_of_transport(
        mean_mechanical_power=mean_p,
        total_mass=float(np.sum(env.model.body_mass)),
        gravity=9.81,
        mean_forward_velocity=mean_v,
    )
    survived = evaluate_survival(has_torso_contact, dist, min_distance_threshold=0.2)

    return {
        "stride_freq": stride_freq,
        "step_height": step_height,
        "stance_depth": stance_depth,
        "kp": kp,
        "kd": kd,
        "distance": dist,
        "mean_v": mean_v,
        "cot": cot,
        "torso_contact": has_torso_contact,
        "survived": survived,
        "steps_completed": len(forward_vels),
    }


def run_bounded_search():
    freqs = [1.5, 2.0, 2.5]
    heights = [0.06, 0.08, 0.10]
    depths = [0.325, 0.340]
    gains = [(55.0, 1.8), (90.0, 3.0)]

    grid = list(itertools.product(freqs, heights, depths, gains))
    print(f"Starting bounded grid search over {len(grid)} candidate configurations...")

    results = []
    for idx, (f, h, d, (kp, kd)) in enumerate(grid):
        res = evaluate_trot_config(f, h, d, kp, kd, duration=5.0)
        results.append(res)
        print(f"[{idx+1:02d}/{len(grid)}] f={f:.1f}Hz h={h:.2f}m d={d:.3f}m kp={kp:.0f} kd={kd:.1f} | dist={res['distance']:+.3f}m mean_v={res['mean_v']:+.3f}m/s survived={res['survived']}")

    # Sort by distance traveled (forward progress)
    results_sorted = sorted(results, key=lambda r: r["distance"], reverse=True)
    best = results_sorted[0]
    print("\n--- BEST CONFIGURATION FOUND ---")
    print(best)
    return best, results_sorted


if __name__ == "__main__":
    run_bounded_search()

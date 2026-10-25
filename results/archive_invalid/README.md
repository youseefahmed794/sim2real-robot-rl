# Archive of Invalid Previous Results

These results were produced prior to the validity audit:
- Produced while `mujoco_env.py` was being edited.
- Flawed evaluation protocol (eval scenarios overlapped with training DR ranges).
- Cost of Transport (COT) was computed with `distance_traveled` clamped to `0.01` m when stationary, producing unphysical COT ~177.5.
- PPO was trained on a nominal budget of only 15,000 steps (~30 episodes), during which the policy learned only to stand still without walking (forward velocity ~0.0 m/s, tracking error 0.50 m/s).
- Classical baseline was untuned and not making forward progress.

DO NOT CITE THESE ARCHIVED RESULTS.

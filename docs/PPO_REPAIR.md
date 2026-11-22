# Shared PPO optimizer repair

## Problem and scope

Published base: 6fd44200c5bda735324ea583be87fadc13d5878f.
The original PPO.train_step evaluated NumPy losses but never performed a gradient
or optimizer update. A behavioral regression observed zero actor parameter change.
The repaired backend uses PyTorch autograd and Adam, clips gradients, and checks
finite objectives. Checkpoints retain model, optimizer, configuration, and RNG state.
Old NumPy checkpoints remain historical inference artifacts.

Only the shared algorithm, selected tests, its dependency declarations, and this
validation/documentation scope are included. Models, rewards, residual control,
heading experiments, and vectorized trainers remain outside this repair.

## Local candidate validation

- 64 tests passed; no failures or skips.
- Actor, critic, and log standard deviation changed in the controlled update test.
- The fixed-batch critic test, KL stop, gradient clipping, and checkpoint continuation passed.
- The contextual bandit used 20480 one-step transitions.
- Initial deterministic held-out MSE: 0.12356419253.
- Final deterministic held-out MSE: 9.76805470098e-05.
- MSE reduction: 99.920948%.
- Optimizer steps: 1116.
- Checkpoint reload produced identical deterministic actions.

Acceptance was MSE below 0.01 and at least 90% reduction. This is one fixed training
seed and one held-out input seed, not a locomotion or hardware transfer result.
The machine-readable [validation record](ppo_repair_validation.json) preserves
source and log hashes plus the observed gate summary. Coverage was not remeasured.

## Reproduce

Use Python 3.12 in an isolated environment. Install PyTorch CPU from its official
package index, then the repository requirements. Run from the repository root:

    python -m pytest -c /dev/null --noconftest -p no:cacheprovider test -v
    PYTHONPATH=. python scripts/prove_ppo_learning.py --output /tmp/ppo-gate-new

Use a fresh output directory. The installed dependency ranges permit multiple
versions; CI records its actual installed versions as an artifact. The observed
local gate used PyTorch 2.14.1+cpu and NumPy 2.5.4.

## Interpretation limits and historical results

Old PPO A/B outputs cannot establish a learning or DR effect, since their backend
did not update weights. Original numerical outputs remain preserved.
Fresh locomotion/DR claims require a new, versioned experiment after collector
and evaluation audits. The existing report generators now emit neutral
interpretation guidance rather than hard-coded learning conclusions.
Shared-buffer truncation support does not certify all legacy caller behavior.

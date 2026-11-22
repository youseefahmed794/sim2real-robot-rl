"""Cheap contextual bandit gate for the shared production PPO class.

Known optimum: a*(x) = 0.5*x + 0.2. Reward = -(clip(a,-1,1)-a*)**2.
Acceptance declared before training: held-out deterministic MSE < .01 and
at least 90% reduction from initialization. This is not a locomotion result.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sim2real_rl.algo.network import PyTorchActorCritic
from sim2real_rl.algo.ppo import PPO, PPOConfig
from sim2real_rl.algo.buffer import RolloutBuffer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/ppo_learning_gate')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    acceptance = {'heldout_mse_below': 0.01, 'mse_reduction_at_least': 0.90}
    print('Predeclared acceptance: '+json.dumps(acceptance), flush=True)
    torch.set_num_threads(1)
    torch.manual_seed(20261026)
    np.random.seed(20261026)
    train_rng = np.random.default_rng(20261026)
    heldout = np.random.default_rng(20261027).uniform(-1, 1, (1024, 1))
    agent = PPO(PyTorchActorCritic(1, 1, [16, 16], init_std=0.5),
                PPOConfig(num_epochs=4, mini_batch_size=64, learning_rate=0.003,
                          entropy_coef=0, target_kl=0.02))
    before = {k:v.detach().clone() for k,v in agent.actor_critic.state_dict().items()}

    def evaluate():
        actions, _, _ = agent.actor_critic.act(heldout, deterministic=True)
        return float(np.mean((np.clip(actions,-1,1) - (0.5*heldout+0.2))**2))

    initial_mse = evaluate()
    traces = []
    for iteration in range(80):
        buf = RolloutBuffer(256, 1, 1)
        for x in train_rng.uniform(-1, 1, (256, 1)):
            action, logp, value = agent.actor_critic.act(x)
            reward = -float((np.clip(action[0],-1,1)-(0.5*x[0]+0.2))**2)
            buf.add(x, action, reward, True, value, logp)
        buf.compute_returns_and_advantages()
        metrics = agent.train_step(buf)
        traces.append(dict(iteration=iteration+1, mean_rollout_reward=float(buf.rewards.mean()),
                           heldout_mse=evaluate(), **metrics))
    final_mse = evaluate()
    deltas = {g: float(torch.sqrt(sum((v-before[k]).square().sum()
                    for k,v in agent.actor_critic.state_dict().items() if k.startswith(g))))
              for g in ('actor','critic','log_std')}
    agent.save(output/'bandit.pt')
    restored = PPO()
    restored.load(output/'bandit.pt')
    a = agent.actor_critic.act(heldout, deterministic=True)[0]
    b = restored.actor_critic.act(heldout, deterministic=True)[0]
    identical_reload = np.array_equal(a,b)
    passed = final_mse < acceptance['heldout_mse_below'] and final_mse < .1*initial_mse and identical_reload
    report = dict(task='one-step continuous contextual bandit', seed=20261026,
                  heldout_seed=20261027, heldout_inputs=1024, transitions=80*256,
                  acceptance=acceptance, initial_mse=initial_mse, final_mse=final_mse,
                  mse_reduction=1-final_mse/initial_mse, parameter_deltas_l2=deltas,
                  optimizer_steps=agent.update_count, identical_checkpoint_reload=identical_reload,
                  passed=passed, torch_version=torch.__version__, numpy_version=np.__version__, traces=traces)
    (output/'bandit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'traces'}, indent=2, allow_nan=False))
    if not passed:
        raise SystemExit('Learning gate failed')


if __name__ == '__main__':
    main()

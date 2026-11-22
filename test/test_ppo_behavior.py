"""Behavioral gates for the shared production PPO training path."""
import numpy as np
from sim2real_rl.algo.buffer import RolloutBuffer
from sim2real_rl.algo.ppo import PPO, PPOConfig


def on_policy_batch(agent, n=32):
    obs = np.random.default_rng(20261026).normal(size=(n, agent.actor_critic.observation_dim))
    buf = RolloutBuffer(n, obs.shape[1], agent.actor_critic.action_dim)
    for x in obs:
        a, lp, v = agent.actor_critic.act(x)
        target = 0.4 + 0.2 * x[0]
        reward = -float(np.mean((np.clip(a, -1, 1) - target) ** 2))
        buf.add(x, a, reward, True, v, lp)
    buf.compute_returns_and_advantages()
    return buf


def flatten(weights):
    return {**{f'{group}.{kind}.{i}': a.copy()
               for group in ('actor', 'critic') for kind in ('weights', 'biases')
               for i, a in enumerate(weights[group][kind])},
            'log_std': weights['log_std'].copy()}


def test_on_policy_updates_actor_critic_and_log_std():
    import torch
    torch.manual_seed(20261026)
    np.random.seed(20261026)
    agent = PPO(config=PPOConfig(num_epochs=4, mini_batch_size=8, target_kl=None))
    buf = on_policy_batch(agent)
    before = flatten(agent.actor_critic.get_weights())
    metrics = agent.train_step(buf)
    after = flatten(agent.actor_critic.get_weights())
    for group in ('actor', 'critic', 'log_std'):
        delta = np.linalg.norm(np.concatenate([(after[k]-v).ravel() for k,v in before.items() if k.startswith(group)]))
        assert delta > 1e-6, (group, delta)
    for group in ('actor', 'critic', 'log_std'):
        assert np.isfinite(metrics[f'{group}_grad_norm'])
        assert metrics[f'{group}_grad_norm'] > 0


def small_agent(**kwargs):
    import torch
    from sim2real_rl.algo.network import PyTorchActorCritic
    torch.manual_seed(20261026)
    np.random.seed(20261026)
    return PPO(PyTorchActorCritic(3, 1, [16, 16]), PPOConfig(**kwargs))


def test_fixed_batch_critic_fit():
    agent = small_agent(num_epochs=4, mini_batch_size=32, target_kl=None, learning_rate=0.003, entropy_coef=0)
    buf = on_policy_batch(agent)
    # Isolate supervised value regression on fixed, real one-step returns.
    buf.advantages[:] = 0
    before = np.mean((agent.actor_critic.get_value(buf.observations)-buf.returns)**2)
    for _ in range(120):
        agent.train_step(buf)
    after = np.mean((agent.actor_critic.get_value(buf.observations)-buf.returns)**2)
    assert after < 0.1 * before, (before, after)


def test_checkpoint_resumes_exact_adam_update(tmp_path):
    import torch
    agent = small_agent(num_epochs=2, mini_batch_size=8, target_kl=None)
    agent.train_step(on_policy_batch(agent))
    checkpoint = tmp_path/'resume.pt'
    agent.save(checkpoint)
    old_weights = flatten(agent.actor_critic.get_weights())
    # Exercise stochastic rollout RNG and minibatch permutation after checkpoint.
    batch = on_policy_batch(agent)
    result = agent.train_step(batch)
    resumed = small_agent()
    resumed.load(checkpoint)
    assert resumed.update_count == 8
    assert resumed.optimizer.state_dict()['state']
    assert resumed.config == agent.config
    for k,v in flatten(resumed.actor_critic.get_weights()).items():
        np.testing.assert_array_equal(v, old_weights[k])
    replay_batch = on_policy_batch(resumed)
    np.testing.assert_array_equal(batch.actions, replay_batch.actions)
    assert resumed.train_step(replay_batch) == result
    for k,v in agent.actor_critic.state_dict().items():
        torch.testing.assert_close(v, resumed.actor_critic.state_dict()[k], rtol=0, atol=0)
    assert agent.update_count == resumed.update_count
    for k,v in agent.optimizer.state_dict()['state'].items():
        for key, tensor in v.items():
            torch.testing.assert_close(tensor, resumed.optimizer.state_dict()['state'][k][key], rtol=0, atol=0)


def test_kl_stop_skips_excessive_update():
    agent = small_agent(num_epochs=4, mini_batch_size=8, target_kl=0.001)
    buf = on_policy_batch(agent)
    # Deliberately stale rollout to trigger the safety gate before optimizer.step.
    buf.log_probs[:] += 2
    before = flatten(agent.actor_critic.get_weights())
    metrics = agent.train_step(buf)
    assert metrics['early_stopped']
    assert metrics['optimizer_steps'] == 0
    for k,v in flatten(agent.actor_critic.get_weights()).items():
        np.testing.assert_array_equal(v, before[k])


def test_numpy_inference_backend_rejected():
    import pytest
    from sim2real_rl.algo.network import ActorCritic
    with pytest.raises(TypeError, match='inference-only'):
        PPO(ActorCritic(3, 1, [8]))


def test_std_floor_and_latent_action_likelihood():
    import torch
    agent = small_agent()
    model = agent.actor_critic
    obs = np.zeros((32, 3))
    actions, logp, _ = model.act(obs)
    # Store raw latent samples, including samples outside environment bounds.
    assert np.any(np.abs(actions) > 1)
    recomputed, _, _ = model.evaluate_actions(obs, actions)
    np.testing.assert_allclose(recomputed.detach().numpy(), logp, atol=1e-6)
    with torch.no_grad():
        model.log_std.fill_(np.log(model.min_std)-1)
    _, _, entropy = model.evaluate_actions(obs, actions)
    entropy.backward()
    torch.testing.assert_close(model.log_std.grad, torch.zeros_like(model.log_std))
    np.testing.assert_allclose(model.std.detach().numpy(), model.min_std)


def test_gradient_clipping():
    import torch
    agent = small_agent(num_epochs=1, mini_batch_size=32, target_kl=None, max_grad_norm=0.01)
    metrics = agent.train_step(on_policy_batch(agent))
    norm = torch.sqrt(sum(p.grad.square().sum() for p in agent.actor_critic.parameters()))
    assert metrics['grad_norm'] > 0.01
    assert float(norm) <= 0.010001


def test_actual_train_agent_entrypoint_uses_trainable_backend(monkeypatch):
    """Run its collection loop on a controlled stub, without an ablation run."""
    import runpy
    from pathlib import Path
    import pytest
    import torch
    from types import SimpleNamespace
    pytest.importorskip('mujoco')
    script = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/run_reproducible_ablation.py'))

    class BanditEnv:
        def __init__(self, **kwargs):
            self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002),
                                         geom_friction=np.ones((1,3)), body_mass=np.ones(2))
            self.decimation = 4
            self.rng = np.random.default_rng(123)
        def reset(self, **kwargs):
            return self.rng.normal(size=48), {}
        def step(self, action):
            # End each controlled episode by time limit, to exercise the caller.
            return self.rng.normal(size=48), -float(np.mean((action-0.3)**2)), False, True, {}

    function = script['train_agent']
    monkeypatch.setitem(function.__globals__, 'MuJoCoQuadrupedEnv', BanditEnv)
    torch.manual_seed(20261026)
    initial = PPO().actor_critic.get_weights()
    agent, curve = function(20261026, False, {}, total_steps=4, rollout_len=32)
    assert agent.update_count > 0
    assert len(curve) == 4
    assert np.linalg.norm(agent.actor_critic.get_weights()['critic']['weights'][-1] - initial['critic']['weights'][-1]) > 1e-6

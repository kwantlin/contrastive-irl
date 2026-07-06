#!/usr/bin/env python3
"""
Utility for collecting offline goal-conditioned trajectories from a trained CRL policy.

This script mirrors the rollout logic in `notebooks/eval-ant.py`, but stores the
resulting trajectories into a NumPy `.npz` archive that can be consumed by
`src/train_offline_fb.py` for offline FB training.
"""

import argparse
import json
import os
import pickle
from typing import Tuple

import flax.linen as nn
import jax
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from brax.io import model
from brax.training import distribution
from flax.core import FrozenDict
from jax import numpy as jnp
import numpy as np

from utils import get_env_config, create_env


class Net(nn.Module):
    """MLP with residual connections, matching the architecture used in CRL training."""

    output_size: int
    width: int = 1024
    num_blocks: int = 4
    block_size: int = 2
    use_ln: bool = True

    @nn.compact
    def __call__(self, x):
        lecun_uniform = nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")
        normalize = nn.LayerNorm() if self.use_ln else (lambda y: y)
        residual_stream = jnp.zeros((x.shape[0], self.width))
        for _ in range(self.num_blocks):
            for _ in range(self.block_size):
                x = nn.swish(normalize(nn.Dense(self.width, kernel_init=lecun_uniform)(x)))
            x += residual_stream
            residual_stream = x
        x = nn.Dense(self.output_size, kernel_init=lecun_uniform)(x)
        return x


def make_policy(actor: Net, action_dist: distribution.NormalTanhDistribution, params: FrozenDict, deterministic: bool = False):
    """Wraps the actor parameters into a callable policy."""

    def policy(obs, key_sample):
        logits = actor.apply(params, obs[None, ...])
        if deterministic:
            action = action_dist.mode(logits)
        else:
            action = action_dist.sample(logits, key_sample)
        return action[0], {}

    return policy


def collect_rollout(env, policy_fn, rng: jax.Array, episode_length: int) -> Tuple[jax.Array, ...]:
    """Collect a single trajectory of length `episode_length` from `policy_fn`."""

    def step_fn(carry, _):
        state, key = carry
        act_key, next_key = jax.random.split(key)
        action, _ = policy_fn(state.obs, act_key)
        next_state = env.step(state, action)
        transition = (
            state.obs,
            action,
            next_state.reward,
            next_state.done,
        )
        return (next_state, next_key), transition

    init_state = env.reset(rng)
    (_, _), (observations, actions, rewards, dones) = jax.lax.scan(
        step_fn, (init_state, rng), None, length=episode_length
    )
    return observations, actions, rewards, dones


def parse_args():
    parser = argparse.ArgumentParser(description="Collect offline CRL trajectories.")
    parser.add_argument("--run_dir", required=True, help="Path to the CRL run directory (contains args.pkl and ckpt/).")
    parser.add_argument("--ckpt_name", default="best.pkl", help="Checkpoint filename under run_dir/ckpt/.")
    parser.add_argument("--output_path", required=True, help="Path to the output .npz file.")
    parser.add_argument("--num_envs", type=int, default=512, help="Number of rollouts (vmap batch size).")
    parser.add_argument("--episode_length", type=int, default=1024, help="Number of steps per trajectory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for rollout sampling.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = os.path.join(args.run_dir, "ckpt", args.ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    with open(os.path.join(args.run_dir, "args.pkl"), "rb") as f:
        train_args = pickle.load(f)

    config = get_env_config(train_args)
    env = create_env(env_name=train_args.env_name, backend=train_args.backend)
    env = EpisodeWrapper(env, args.episode_length, config.action_repeat)
    env = AutoResetWrapper(env)

    obs_size = env.observation_size
    action_size = env.action_size

    block_size = 2
    num_blocks = max(1, train_args.n_hidden // block_size)
    actor = Net(action_size * 2, train_args.h_dim, num_blocks, block_size, train_args.use_ln)
    params = model.load_params(ckpt_path)
    policy_params, _, _ = params

    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    policy_fn = make_policy(actor, parametric_action_distribution, policy_params, deterministic=args.deterministic)
    jit_policy = jax.jit(policy_fn)

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    def rollout_with_jit(rng):
        def step_fn(carry, _):
            state, key = carry
            act_key, next_key = jax.random.split(key)
            action, _ = jit_policy(state.obs, act_key)
            next_state = jit_env_step(state, action)
            transition = (
                state.obs,
                action,
                next_state.reward,
                next_state.done,
            )
            return (next_state, next_key), transition

        init_state = jit_env_reset(rng)
        (_, _), (observations, actions, rewards, dones) = jax.lax.scan(
            step_fn, (init_state, rng), None, length=args.episode_length
        )
        return observations, actions, rewards, dones

    rollout_fn = jax.jit(rollout_with_jit)
    rng = jax.random.PRNGKey(args.seed)
    episode_rngs = jax.random.split(rng, args.num_envs)
    observations, actions, rewards, dones = jax.vmap(rollout_fn)(episode_rngs)

    metadata = {
        "env_name": train_args.env_name,
        "backend": train_args.backend,
        "state_dim": int(env.state_dim),
        "goal_dim": int(obs_size - env.state_dim),
        "obs_dim": int(obs_size),
        "action_dim": int(action_size),
        "episode_length": int(args.episode_length),
        "num_envs": int(args.num_envs),
        "run_dir": args.run_dir,
        "ckpt_name": args.ckpt_name,
    }

    traj_ids = jnp.tile(jnp.arange(args.num_envs)[:, None], (1, args.episode_length)).astype(jnp.float32)

    np.savez_compressed(
        args.output_path,
        observations=np.asarray(observations),
        actions=np.asarray(actions),
        rewards=np.asarray(rewards),
        dones=np.asarray(dones),
        traj_ids=np.asarray(traj_ids),
        metadata=json.dumps(metadata),
    )
    print(f"Saved dataset with shape {observations.shape} to {args.output_path}")


if __name__ == "__main__":
    main()


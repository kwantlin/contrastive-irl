import argparse
from datetime import datetime
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.getcwd())

from envs.ant_base import AntForward, AntJump, AntFlip
from envs.walker2d import WalkerForward, WalkerJump, WalkerFlip

import brax
from brax import envs
from brax.io import html
from brax.io import model
from brax.training.agents.ppo import train as ppo_train
from brax.training.agents.ppo import networks as ppo_networks

import jax

# Register custom environments
envs.register_environment('antforward', AntForward)
envs.register_environment('antjump', AntJump)
envs.register_environment('antflip', AntFlip)
envs.register_environment('walkerforward', WalkerForward)
envs.register_environment('walkerjump', WalkerJump)
envs.register_environment('walkerflip', WalkerFlip)

def main(args):
  """Main training function."""
  # Environment
  if args.env == 'antforward':
    env = AntForward(min_forward_velocity=args.min_forward_velocity)
  elif args.env == 'antjump':
    env = AntJump(target_jump_height=args.target_jump_height)
  elif args.env == 'walkerforward':
    env = WalkerForward(min_forward_velocity=args.min_forward_velocity)
  elif args.env == 'walkerjump':
    env = WalkerJump(min_jump_height=args.target_jump_height)
  elif args.env == 'walkerflip':
    env = WalkerFlip(min_flip_velocity=args.min_flip_velocity)
  elif args.env == 'antflip':
    env = AntFlip(min_flip_velocity=args.min_flip_velocity)

  print(args.env)
  # Create a string for filenames based on parameters
  param_str = ''
  if 'forward' in args.env:
    param_str = f'_vel{args.min_forward_velocity}'
  elif 'jump' in args.env:
    param_str = f'_h{args.target_jump_height}'
  elif 'flip' in args.env:
    param_str = f'_flipvel{args.min_flip_velocity}'

  print(param_str)
  # PPO network factory
  network_factory = ppo_networks.make_ppo_networks

  times = [datetime.now()]

  def progress(num_steps, metrics):
    times.append(datetime.now())
    print(
        '  Steps: {:,}, Time: {}, Eval Mean Reward: {:,.4f}'.format(
            num_steps, times[-1] - times[0], metrics['eval/episode_reward']
        )
    )

  print(f'Training {args.env} with PPO...')
  # PPO train function
  make_policy, params, _ = ppo_train.train(
      environment=env,
      num_timesteps=args.total_env_steps,
      episode_length=args.episode_length,
      num_envs=args.num_envs,
      learning_rate=args.lr,
      entropy_cost=args.entropy_cost,
      discounting=args.discounting,
      seed=args.seed,
      unroll_length=args.unroll_length,
      batch_size=args.batch_size,
      num_minibatches=args.num_minibatches,
      num_updates_per_batch=args.num_updates_per_batch,
      num_evals=args.num_evals,
      normalize_observations=True,
      network_factory=network_factory,
      progress_fn=progress,
  )
  print('Training finished.')

  
  # Save model
  model_path = f'ppo_{args.env}{param_str}_model.pkl'
  model.save_params(model_path, params)
  print(f'Model saved to {model_path}')

  # Visualize
  print('Creating video...')
  policy = make_policy(params, deterministic=True)

  jit_env_reset = jax.jit(env.reset)
  jit_env_step = jax.jit(env.step)
  jit_policy = jax.jit(policy)

  rollouts = []
  rng = jax.random.PRNGKey(args.seed)

  for i in range(args.num_eval_episodes):
    print(f'Visualizing episode {i+1}/{args.num_eval_episodes}')
    rng, reset_rng = jax.random.split(rng)
    state = jit_env_reset(rng=reset_rng)
    rollout = [state.pipeline_state]
    for _ in range(args.episode_length):
      act_rng, rng = jax.random.split(rng)
      act, _ = jit_policy(state.obs, act_rng)
      state = jit_env_step(state, act)
      rollout.append(state.pipeline_state)
      if state.done.all():
        break

    html_path = f'ppo_{args.env}{param_str}_video_{i}.html'
    html.save(html_path, env.sys.tree_replace({'opt.timestep': env.dt}), rollout)
    print(f'Video saved to {html_path}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train PPO on Ant environments.')

  parser.add_argument(
      '--env',
      type=str,
      default='antjump',
      choices=['antforward', 'antjump', 'walkerforward', 'walkerjump', 'walkerflip', 'antflip'],
      help='Environment to train.',
  )
  parser.add_argument(
      '--total_env_steps',
      type=int,
      default=10_000_000,
      help='Total number of environment steps to train for.',
  )
  parser.add_argument(
      '--episode_length', type=int, default=1000, help='Length of each episode.'
  )
  parser.add_argument(
      '--num_envs',
      type=int,
      default=2048,
      help='Number of parallel environments.',
  )
  parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
  parser.add_argument(
      '--entropy_cost',
      type=float,
      default=3e-4,
      help='Entropy cost coefficient.',
  )
  parser.add_argument(
      '--discounting', type=float, default=0.97, help='Discounting factor.'
  )
  parser.add_argument(
      '--unroll_length', type=int, default=10, help='Unroll length for PPO.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=1024,
      help='Batch size for PPO updates.',
  )
  parser.add_argument(
      '--num_minibatches',
      type=int,
      default=8,
      help='Number of minibatches for PPO updates.',
  )
  parser.add_argument(
      '--num_updates_per_batch',
      type=int,
      default=8,
      help='Number of updates per batch for PPO.',
  )
  parser.add_argument(
      '--num_evals',
      type=int,
      default=20,
      help='Number of evaluations during training.',
  )
  parser.add_argument('--seed', type=int, default=0, help='Random seed.')
  parser.add_argument(
      '--num_eval_episodes',
      type=int,
      default=5,
      help='Number of episodes for final evaluation and visualization.',
  )
  parser.add_argument(
      '--min_forward_velocity',
      type=float,
      default=0.5,
      help='Minimum forward velocity for forward environments.',
  )
  parser.add_argument(
      '--target_jump_height',
      type=float,
      default=1.0,
      help='Target jump height for jump environments.',
  )
  parser.add_argument(
      '--min_flip_velocity',
      type=float,
      default=1.0,
      help='Minimum angular velocity for flip environments.',
  )

  args = parser.parse_args()
  main(args) 
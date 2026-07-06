import functools
import time
from typing import Callable, Optional, NamedTuple

import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
from jax import numpy as jnp
import optax
import numpy as np
from absl import logging
import brax
from brax import envs
from brax.training import gradients, distribution, types, pmap
from brax.training.replay_buffers_test import jit_wrap

from envs.wrappers import TrajectoryIdWrapper
from src.evaluator import CrlEvaluator
from src.replay_buffer import ReplayBufferState, Transition, TrajectoryUniformSamplingQueue

Metrics = types.Metrics
Env = envs.Env
State = envs.State
_PMAP_AXIS_NAME = "i"

# The SAEncoder, GoalEncoder, and Actor all use the same function. Output size for SA/Goal encoders should be representation size, and for Actor should be 2 * action_size.
# To keep parity with the existing architecture, by default we only use one residual block of depth 2, hence effectively not using the residual connections.
class Net(nn.Module):
    """
    MLP with residual connections: residual blocks have $block_size layers. Uses swish activation, optionally uses layernorm.
    """
    output_size: int
    width: int = 1024
    num_blocks: int = 1
    block_size: int = 2
    use_ln: bool = True
    @nn.compact
    def __call__(self, x):
        lecun_uniform = nn.initializers.variance_scaling(1/3, "fan_in", "uniform")
        normalize = nn.LayerNorm() if self.use_ln else (lambda x: x)
        
        # Start of net
        residual_stream = jnp.zeros((x.shape[0], self.width))
        
        # Main body
        for i in range(self.num_blocks):
            for j in range(self.block_size):
                x = nn.swish(normalize(nn.Dense(self.width, kernel_init=lecun_uniform)(x)))
            x += residual_stream
            residual_stream = x
                
        # Last layer mapping to representation dimension
        x = nn.Dense(self.output_size, kernel_init=lecun_uniform)(x)
        return x

# The brax version of this does not take in the actor and action_distribution arguments; before we pass it to brax evaluator or return it from train(), we do a partial application.
def make_policy(actor, parametric_action_distribution, backward_repr, params, state_dim, repr_dim, deterministic=False):
    actor_params, backward_repr_params = params
    def policy(obs, key_sample):
        state = obs[:, :state_dim]
        goal = obs[:, state_dim:]
        goal_repr = backward_repr.apply(backward_repr_params, goal)
        goal_repr = goal_repr / jnp.linalg.norm(goal_repr, axis=-1, keepdims=True) * jnp.sqrt(repr_dim)
        policy_obs = jnp.concatenate([state, goal_repr], axis=-1)
        logits = actor.apply(actor_params, policy_obs)
        if deterministic:
            action = parametric_action_distribution.mode(logits)
        else:
            action = parametric_action_distribution.sample(logits, key_sample)
            print("ACTION SHAPE", action.shape)
        extras = {}
        return action, extras
    return policy

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    actor_state: TrainState
    # critic_state: TrainState
    # value_state: TrainState
    fb_repr_state: TrainState
    # target_critic_params: flax.core.FrozenDict
    target_forward_repr_params: flax.core.FrozenDict
    target_backward_repr_params: flax.core.FrozenDict

def _init_training_state(key, actor, forward_repr, backward_repr, state_dim, goal_dim, action_dim, repr_dim, episode_length, actor_lr, repr_lr, num_local_devices_to_use):
    """
    Initializes the training state for a forward-backward representation learning model.
    """
    actor_key, critic_key, value_key, forward_key, backward_key = jax.random.split(key, 5)
    
    # Actor
    actor_params = actor.init(actor_key, jnp.ones([1, state_dim + repr_dim]))
    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(learning_rate=actor_lr))

    # Critic and Value
    # critic_params = critic.init(critic_key, jnp.ones([1, state_dim + action_dim + goal_dim]))
    # value_params = value.init(value_key, jnp.ones([1, state_dim + goal_dim]))
    # critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optax.adam(learning_rate=critic_lr))
    # value_state = TrainState.create(apply_fn=value.apply, params=value_params, tx=optax.adam(learning_rate=critic_lr))

    # Forward and backward representation networks
    forward_repr_params = forward_repr.init(forward_key, jnp.ones([1, state_dim + action_dim + repr_dim]))
    backward_repr_params = backward_repr.init(backward_key, jnp.ones([1, goal_dim]))
    
    # Single optimizer for both forward and backward repr networks
    repr_optimizer = optax.adam(learning_rate=repr_lr)
    fb_repr_state = TrainState.create(apply_fn=None, params=(forward_repr_params, backward_repr_params), tx=repr_optimizer)

    # Target networks: just store params
    # target_critic_params = jax.tree_util.tree_map(lambda x: x.copy(), critic_params)
    target_forward_repr_params = jax.tree_util.tree_map(lambda x: x.copy(), forward_repr_params)
    target_backward_repr_params = jax.tree_util.tree_map(lambda x: x.copy(), backward_repr_params)

    training_state = TrainingState(
        env_steps=jnp.zeros(()), 
        gradient_steps=jnp.zeros(()), 
        actor_state=actor_state,
        # critic_state=critic_state,
        # value_state=value_state,
        fb_repr_state=fb_repr_state,
        # target_critic_params=target_critic_params,
        target_forward_repr_params=target_forward_repr_params,
        target_backward_repr_params=target_backward_repr_params
    )
    
    training_state = jax.device_put_replicated(training_state, jax.local_devices()[:num_local_devices_to_use])
    return training_state


def forward_backward_repr_loss(
    forward_params, backward_params, forward_repr, backward_repr, transitions, state_dim, goal_dim, goal_indices, repr_dim, key,
    training_state, actor, parametric_action_distribution, repr_agg='mean', orthonorm_coef=1.0, discount=0.99
):
    """
    Compute the forward-backward representation loss, following the structure of ogbench/impls/agents/fb_repr.py.
    """
    states = transitions.observation[:, :state_dim]
    actions = transitions.action
    next_states = transitions.extras["next_state"][:, :state_dim]
    print("fb: next_states shape", next_states.shape)
    print("fb: next_states[:, goal_indices] shape", next_states[:, goal_indices].shape)
    goals = transitions.observation[:, state_dim:]
    print("fb: goals shape", goals.shape)
    batch_size = states.shape[0]
    key, key_sample = jax.random.split(key)
    gaussian_latents = sample_latents(batch_size, repr_dim, key_sample)
    # Permute goals before computing backward-derived latents to break identity coupling
    key, key_perm = jax.random.split(key)
    perm = jax.random.permutation(key_perm, batch_size)
    permuted_goals = goals[perm]
    future_latents = backward_repr.apply(jax.lax.stop_gradient(training_state.fb_repr_state.params[1]), permuted_goals)
    future_latents = future_latents / jnp.linalg.norm(future_latents, axis=-1, keepdims=True) * jnp.sqrt(repr_dim)
    # Generate random coin flips for each batch element
    key, key_flip = jax.random.split(key)
    coin_flips = jax.random.bernoulli(key_flip, shape=(batch_size,), p=0.5)
    
    # Use coin flips to select between gaussian and future latents
    latents = jnp.where(
        coin_flips[:, None],  # Expand dims to match latent shape
        gaussian_latents,     # If True (p=0.5)
        future_latents        # If False (p=0.5) 
    )
    print("fb: latents shape", latents.shape)
    
    batch_size = states.shape[0]
    latent_dim = repr_dim

    # Compute next actions using the actor
    next_dist = actor.apply(jax.lax.stop_gradient(training_state.actor_state.params), jnp.concatenate([next_states, latents], axis=-1))
    key, subkey = jax.random.split(key)
    next_actions = parametric_action_distribution.sample(next_dist, subkey)

    # Compute target forward and backward representations using target params
    next_forward_reprs_all = forward_repr.apply(training_state.target_forward_repr_params, jnp.concatenate([next_states, next_actions, latents], axis=-1))
    next_backward_reprs = backward_repr.apply(training_state.target_backward_repr_params, next_states[:, goal_indices])
    next_backward_reprs = next_backward_reprs / jnp.linalg.norm(next_backward_reprs, axis=-1, keepdims=True) * jnp.sqrt(repr_dim)
    
    # Split two forward heads and compute target occupancy for each, then mean across heads
    next_f1, next_f2 = jnp.split(next_forward_reprs_all, 2, axis=-1)
    target_occ_measures_heads = jnp.stack([
        jnp.einsum('bd,td->bt', next_f1, next_backward_reprs),
        jnp.einsum('bd,td->bt', next_f2, next_backward_reprs)
    ], axis=0)  # (2, B, B)
    target_occ_measures = jnp.mean(target_occ_measures_heads, axis=0)  # (B, B)

    # target_occ_measures is already aggregated across heads to shape (B, B)

    # Compute current forward and backward representations
    forward_reprs_all = forward_repr.apply(forward_params, jnp.concatenate([states, actions, latents], axis=-1))
    backward_reprs = backward_repr.apply(backward_params, next_states[:, goal_indices])
    backward_reprs = backward_reprs / jnp.linalg.norm(backward_reprs, axis=-1, keepdims=True) * jnp.sqrt(repr_dim)
    f1, f2 = jnp.split(forward_reprs_all, 2, axis=-1)
    occ_measures_heads = jnp.stack([
        jnp.einsum('bd,td->bt', f1, backward_reprs),
        jnp.einsum('bd,td->bt', f2, backward_reprs)
    ], axis=0)  # (2, B, B)

    print("fb: occ_measures shape", occ_measures_heads.shape)
    print("fb: target_occ_measures shape", target_occ_measures.shape)
    I = jnp.eye(occ_measures_heads.shape[-1])
    x = occ_measures_heads - discount * target_occ_measures  # broadcast (2,B,B)-(B,B)
    repr_off_diag_loss = ((x * (1 - I)) ** 2)
    repr_off_diag_loss = 0.5 * jnp.sum(repr_off_diag_loss, axis=-1) / (occ_measures_heads.shape[-1] - 1)  # (2, B)

    repr_diag_loss = -jnp.diagonal(occ_measures_heads, axis1=-2, axis2=-1)  # (2, B)

    repr_loss = jnp.mean(
        repr_diag_loss + repr_off_diag_loss
    )

    # Orthonormalization loss
    covariance = jnp.matmul(backward_reprs, backward_reprs.T)
    ortho_diag_loss = -2 * jnp.diag(covariance)
    ortho_off_diag_loss = (covariance * (1 - I)) ** 2
    ortho_loss = orthonorm_coef * jnp.mean(
        ortho_diag_loss + jnp.sum(ortho_off_diag_loss, axis=-1) / (occ_measures_heads.shape[-1] - 1)
    )

    total_loss = repr_loss + ortho_loss

    metrics = {
        'repr_loss': repr_loss,
        # Mean across both heads and batch
        'repr_diag_loss': jnp.mean(repr_diag_loss),
        'repr_off_diag_loss': jnp.mean(repr_off_diag_loss),
        'ortho_loss': ortho_loss,
        'ortho_diag_loss': jnp.mean(ortho_diag_loss),
        'ortho_off_diag_loss': jnp.mean(jnp.sum(ortho_off_diag_loss, axis=-1) / (occ_measures_heads.shape[-1] - 1)),
        'occ_measure_mean': occ_measures_heads.mean(),
        'occ_measure_max': occ_measures_heads.max(),
        'occ_measure_min': occ_measures_heads.min(),
    }

    return total_loss, metrics

def actor_loss(actor_params, training_state, actor, forward_repr, backward_repr, parametric_action_distribution, transitions, state_dim, goal_dim, repr_dim, key, entropy_coef=0.1):
    """Compute the FB-style actor loss (not IQL)."""
    states = transitions.observation[:, :state_dim]
    goals = transitions.observation[:, state_dim:]
    print("fb: goals before backward repr", goals.shape)

    goals = jax.lax.stop_gradient(backward_repr.apply(training_state.fb_repr_state.params[1], goals))
    goals = goals / jnp.linalg.norm(goals, axis=-1, keepdims=True) * jnp.sqrt(repr_dim)
    print("fb: goals after backward repr", goals.shape)
    # Sample actions from the actor
    action_mean_and_SD = actor.apply(actor_params, jnp.concatenate([states, goals], axis=-1))
    actions = parametric_action_distribution.sample(action_mean_and_SD, key)

    # Compute forward representations for these actions (two heads) and use mean
    F_all = forward_repr.apply(training_state.fb_repr_state.params[0], jnp.concatenate([states, actions, goals], axis=-1))
    F1, F2 = jnp.split(F_all, 2, axis=-1)
    Q1 = jnp.einsum('sd,sd->s', F1, goals)
    Q2 = jnp.einsum('sd,sd->s', F2, goals)
    Q = 0.5 * (Q1 + Q2)

    # Actor loss: negative mean Q-value across heads
    actor_loss = -Q

    # Entropy regularization if Gaussian
    # log_prob = parametric_action_distribution.log_prob(action_mean_and_SD, actions)
    # actor_loss = actor_loss + entropy_coef * log_prob
    # mean_log_prob = log_prob.mean()
    mean_log_prob = 0.0

    actor_loss = actor_loss.mean()

    metrics = {
        'actor_loss': actor_loss,
        'actor_Q': Q.mean(),
        'actor_log_prob': mean_log_prob,
    }
    return actor_loss, metrics

def critic_loss(critic_params, value_params, training_state, critic, value, forward_repr, backward_repr, parametric_action_distribution, transitions, state_dim, goal_dim, repr_dim, key, discount=0.99):
    """Compute the IQL critic loss (matching fb_repr.py logic, using constant discount)."""
    states = transitions.observation[:, :state_dim]
    actions = transitions.action
    goals = transitions.observation[:, state_dim:]
    next_states = transitions.extras["next_state"][:, :state_dim]
    rewards = transitions.reward

    # Compute next_v using value network
    next_v = value.apply(value_params, jnp.concatenate([next_states, goals], axis=-1))

    # Get q1, q2 from critic
    q1 = critic.apply(critic_params, jnp.concatenate([states, actions, goals], axis=-1))

    # Compute target q using the provided discount constant
    q = rewards + discount * next_v

    # Compute critic loss as mean squared error for both q1 and q2
    critic_loss = ((q1 - q) ** 2).mean()

    metrics = {
        'critic_loss': critic_loss,
        'q_mean': q.mean(),
        'q_max': q.max(),
        'q_min': q.min(),
    }
    return critic_loss, metrics

def value_loss(value_params, training_state, value, critic, parametric_action_distribution, transitions, state_dim, goal_dim, repr_dim, key, expectile=0.9):
    """Compute the IQL value loss (matching fb_repr.py logic)."""
    # Unpack states, actions, goals
    states = transitions.observation[:, :state_dim]
    goals = transitions.observation[:, state_dim:]
    actions = transitions.action

    # Compute Q-values from target critic (q1, q2)
    q= critic.apply(training_state.target_critic_params, jnp.concatenate([states, actions, goals], axis=-1))

    # Compute value estimates
    v = value.apply(value_params, jnp.concatenate([states, goals], axis=-1))

    # Expectile loss (as in fb_repr.py)
    diff = q - v
    weight = jnp.where(diff >= 0, expectile, 1 - expectile)
    value_loss = jnp.mean(weight * (diff ** 2))

    metrics = {
        'value_loss': value_loss,
        'v_mean': v.mean(),
        'v_max': v.max(),
        'v_min': v.min(),
    }
    return value_loss, metrics


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)

def sample_latents(batch_size, latent_dim, key):
    """Sample latents by generating random latents only (no mixing with backward representations)."""
    latent_rng = key
    # Random latents
    latents = jax.random.normal(latent_rng, shape=(batch_size, latent_dim))
    latents = latents / jnp.linalg.norm(latents, axis=-1, keepdims=True) * jnp.sqrt(latent_dim)
    return latents


def load_offline_dataset(dataset_path, env, config, seed):
    """Loads a saved numpy dataset of trajectories and converts it to FB transitions."""
    if not dataset_path:
        raise ValueError("dataset_path must be provided for offline training.")
    data = np.load(dataset_path)
    observations = jnp.asarray(data["observations"])
    actions = jnp.asarray(data["actions"])
    rewards = jnp.asarray(data["rewards"])
    dones = jnp.asarray(data["dones"]).astype(jnp.float32)
    traj_ids = jnp.asarray(data.get("traj_ids"))
    if traj_ids is None:
        num_episodes, seq_len = observations.shape[:2]
        traj_ids = jnp.tile(jnp.arange(num_episodes)[:, None], (1, seq_len)).astype(jnp.float32)

    discounts = 1.0 - dones
    transition = Transition(
        observation=observations,
        action=actions,
        reward=rewards,
        discount=discounts,
        extras={
            "policy_extras": {},
            "state_extras": {
                "truncation": dones,
                "traj_id": traj_ids,
            },
        },
    )

    rng = jax.random.PRNGKey(seed)
    keys = jax.random.split(rng, observations.shape[0])
    flatten_fn = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, None, 0, 0))
    flattened = flatten_fn(config, env, transition, keys)
    flattened = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), flattened)
    return flattened


def sample_epoch_batches(dataset, key, num_steps, batch_size, dataset_size):
    """Samples `num_steps` batches of offline transitions (JAX-friendly)."""
    idx = jax.random.randint(key, shape=(num_steps, batch_size), minval=0, maxval=dataset_size)
    return jax.tree_util.tree_map(lambda x: x[idx], dataset)

def fb_repr_loss_fn(
    params, forward_repr, backward_repr, transitions, state_dim, goal_dim, goal_indices, repr_dim, key,
    training_state, actor, parametric_action_distribution, repr_agg, orthonorm_coef, discount
):
    forward_params, backward_params = params
    loss, metrics = forward_backward_repr_loss(
        forward_params, backward_params, forward_repr, backward_repr, transitions, state_dim, goal_dim, goal_indices, repr_dim, key, 
        training_state, actor=actor,
        parametric_action_distribution=parametric_action_distribution,
        repr_agg=repr_agg,
        orthonorm_coef=orthonorm_coef,
        discount=discount
    )
    return loss, metrics

def train(
    environment: envs.Env,
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    policy_lr: float = 1e-4,
    repr_lr: float = 1e-4,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    deterministic_eval: bool = False,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    eval_env: Optional[envs.Env] = None,
    unroll_length: int = 50,
    train_step_multiplier: int = 1,
    config: NamedTuple = None,
    use_ln: bool = False,
    h_dim: int = 256,
    n_hidden: int = 2,
    repr_dim: int = 64,
    tau: float = 0.005,
    dataset_path: Optional[str] = None,
):
    """
    Trains a forward-backward representation agent using an offline dataset of CRL rollouts.

    Legacy arguments `num_envs`, `min_replay_size`, `max_replay_size`, `unroll_length`,
    and `train_step_multiplier` are ignored but kept for API compatibility.
    `num_timesteps` now denotes the total number of gradient updates to run.
    """

    if not dataset_path:
        raise ValueError("dataset_path must be provided for offline FB training.")

    process_id = jax.process_index()
    num_local_devices_to_use = jax.local_device_count()
    device_count = num_local_devices_to_use * jax.process_count()
    logging.info(
        "local_device_count: %s; total_device_count: %s",
        num_local_devices_to_use,
        device_count,
    )

    if batch_size % device_count != 0:
        raise ValueError("batch_size must be divisible by the total device count for pmapped training.")
    per_device_batch_size = batch_size // device_count

    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    env = TrajectoryIdWrapper(environment)
    env = envs.training.wrap(env, episode_length=episode_length, action_repeat=action_repeat)
    unwrapped_env = environment

    state_dim = env.state_dim
    action_size = env.action_size

    # Network functions
    block_size = 2
    num_blocks = max(1, n_hidden // block_size)
    actor = Net(action_size * 2, h_dim, num_blocks, block_size, use_ln)
    forward_repr = Net(repr_dim * 2, h_dim, num_blocks, block_size, use_ln)
    backward_repr = Net(repr_dim, h_dim, num_blocks, block_size, use_ln)
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)

    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)
    training_state = _init_training_state(
        global_key,
        actor,
        forward_repr,
        backward_repr,
        state_dim,
        len(env.goal_indices),
        env.action_size,
        repr_dim,
        episode_length,
        policy_lr,
        repr_lr,
        num_local_devices_to_use,
    )
    del global_key

    offline_dataset = load_offline_dataset(dataset_path, env, config, seed)
    dataset_size = int(offline_dataset.observation.shape[0])
    offline_dataset = jax.device_put_replicated(offline_dataset, jax.local_devices()[:num_local_devices_to_use])

    actor_update = gradients.gradient_update_fn(
        actor_loss, training_state.actor_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )
    fb_repr_update = gradients.gradient_update_fn(
        fb_repr_loss_fn, training_state.fb_repr_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    def update_step(carry, transitions):
        training_state, key = carry
        key, key_fb, key_actor = jax.random.split(key, 3)

        (fb_loss, fb_metrics), fb_repr_params, fb_repr_opt_state = fb_repr_update(
            training_state.fb_repr_state.params,
            forward_repr,
            backward_repr,
            transitions,
            env.state_dim,
            len(env.goal_indices),
            env.goal_indices,
            repr_dim,
            key_fb,
            training_state,
            actor,
            parametric_action_distribution,
            "mean",
            1.0,
            0.99,
            optimizer_state=training_state.fb_repr_state.opt_state,
        )

        (actor_loss_val, actor_metrics), actor_params, actor_optimizer_state = actor_update(
            training_state.actor_state.params,
            training_state,
            actor,
            forward_repr,
            backward_repr,
            parametric_action_distribution,
            transitions,
            env.state_dim,
            len(env.goal_indices),
            repr_dim,
            key_actor,
            optimizer_state=training_state.actor_state.opt_state,
        )

        new_target_forward_repr_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            fb_repr_params[0],
            training_state.target_forward_repr_params,
        )
        new_target_backward_repr_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            fb_repr_params[1],
            training_state.target_backward_repr_params,
        )

        metrics = {
            "fb_loss": fb_loss,
            "actor_loss": actor_loss_val,
        }
        metrics.update(fb_metrics)
        metrics.update(actor_metrics)

        new_training_state = TrainingState(
            env_steps=training_state.env_steps,
            gradient_steps=training_state.gradient_steps + 1,
            actor_state=training_state.actor_state.replace(params=actor_params, opt_state=actor_optimizer_state),
            fb_repr_state=training_state.fb_repr_state.replace(params=fb_repr_params, opt_state=fb_repr_opt_state),
            target_forward_repr_params=new_target_forward_repr_params,
            target_backward_repr_params=new_target_backward_repr_params,
        )
        return (new_training_state, key), metrics

    num_evals_after_init = max(num_evals - 1, 1)
    total_updates = num_timesteps
    num_training_steps_per_epoch = -(-total_updates // num_evals_after_init)

    def training_epoch(training_state, dataset, key):
        key, sample_key = jax.random.split(key)
        key, update_key = jax.random.split(key)
        transitions = sample_epoch_batches(
            dataset,
            sample_key,
            num_training_steps_per_epoch,
            per_device_batch_size,
            dataset_size,
        )
        (training_state, _), metrics = jax.lax.scan(update_step, (training_state, update_key), transitions)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, key, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    training_walltime = 0.0

    def training_epoch_with_timing(training_state, dataset, key):
        nonlocal training_walltime
        t = time.time()
        training_state, key, metrics = training_epoch(training_state, dataset, key)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_time = time.time() - t
        training_walltime += epoch_time
        updates_per_sec = (per_device_batch_size * num_training_steps_per_epoch * device_count) / epoch_time
        metrics = {
            "training/updates_per_sec": updates_per_sec,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return training_state, key, metrics

    if not eval_env:
        eval_env = environment
    eval_env = TrajectoryIdWrapper(eval_env)
    eval_env = envs.training.wrap(eval_env, episode_length=episode_length, action_repeat=action_repeat)
    global make_policy
    make_policy = functools.partial(
        make_policy,
        actor,
        parametric_action_distribution,
        backward_repr,
        state_dim=env.state_dim,
        repr_dim=repr_dim,
    )
    evaluator = CrlEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=key,
    )

    metrics = {}
    if process_id == 0 and num_evals > 1:
        eval_params = _unpmap((training_state.actor_state.params, training_state.fb_repr_state.params[1]))
        metrics = evaluator.run_evaluation(eval_params, training_metrics={})
        logging.info(metrics)
        progress_fn(0, metrics, make_policy, eval_params, unwrapped_env)

    current_step = 0
    for eval_epoch_num in range(num_evals_after_init):
        logging.info("gradient step %s", current_step)
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, num_local_devices_to_use)
        training_state, epoch_keys, training_metrics = training_epoch_with_timing(
            training_state, offline_dataset, epoch_keys
        )
        current_step = int(_unpmap(training_state.gradient_steps))

        if process_id == 0:
            if checkpoint_logdir:
                params = _unpmap(
                    (
                        training_state.actor_state.params,
                        training_state.fb_repr_state.params,
                        training_state.target_forward_repr_params,
                        training_state.target_backward_repr_params,
                    )
                )
                path = f"{checkpoint_logdir}/step_{current_step}.pkl"
                logging.info("Saving checkpoint at %s", path)
                brax.io.model.save_params(path, params)
            eval_params = _unpmap((training_state.actor_state.params, training_state.fb_repr_state.params[1]))
            metrics = evaluator.run_evaluation(eval_params, training_metrics)
            logging.info(metrics)
            progress_fn(current_step, metrics, make_policy, eval_params, unwrapped_env)

    total_steps = int(_unpmap(training_state.gradient_steps))
    logging.info("total gradient steps: %s", total_steps)
    assert total_steps >= num_timesteps

    pmap.assert_is_replicated(training_state)
    pmap.synchronize_hosts()

    params = _unpmap(
        (
            training_state.actor_state.params,
            training_state.fb_repr_state.params,
            training_state.target_forward_repr_params,
            training_state.target_backward_repr_params,
        )
    )
    logging.info("Returning actor, fb_repr, and target params.")
    return (make_policy, params, metrics)

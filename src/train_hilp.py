import functools
import time
from typing import Callable, Optional, NamedTuple, Any

import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
from jax import numpy as jnp
import optax
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


class MetricValueNet(nn.Module):
    """Metric value function parameterized by a shared representation phi.

    This module computes a negative Euclidean distance between latent embeddings
    of observations and goals, which can be used as a dense shaping value:
        v(s, g) = - ||phi(s) - phi(g)||_2
    """

    latent_dim: int
    width: int = 1024
    num_blocks: int = 1
    block_size: int = 2
    use_ln: bool = True

    def setup(self) -> None:
        self.phi = Net(
            output_size=self.latent_dim,
            width=self.width,
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            use_ln=self.use_ln,
        )

    def __call__(self, observations, goals, *, s_is_phi: bool = False, g_is_phi: bool = False, info: bool = False):
        """Return the metric value and optionally the latent embeddings.

        Args:
            observations: Current observations or phi-encoded observations if `is_phi` is True.
            goals: Goal observations or phi-encoded goals if `is_phi` is True.
            is_phi: Whether `observations` and `goals` are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        if s_is_phi:
            phi_s = observations
        else:
            phi_s = self.phi(observations)
        if g_is_phi:
            phi_g = goals
        else:
            phi_g = self.phi(goals)

        l2_dist = jnp.linalg.norm(phi_s - phi_g, axis=-1)
        v = -l2_dist

        if info:
            return v, phi_s, phi_g
        return v

# The brax version of this does not take in the actor and action_distribution arguments; before we pass it to brax evaluator or return it from train(), we do a partial application.
def make_policy(skill_actor, parametric_action_distribution, value, params, state_dim, repr_dim, goal_indices, deterministic=False):
    skill_actor_params, value_params = params
    def policy(obs, key_sample):
        state = obs[:, :state_dim]
        goal = obs[:, state_dim:]
        # Extract goal coordinates from state (goal_indices are relative to state vector)
        state_goal_portion = state[:, goal_indices]
        # Compute goal latent using phi representation: z* = (phi(g) - phi(s)) / ||phi(g) - phi(s)|| * sqrt(repr_dim)
        _, phi_states, phi_goals = value.apply(value_params, state_goal_portion, goal, info=True)
        goal_latents = phi_goals - phi_states
        goal_latents_norm = jnp.linalg.norm(goal_latents, axis=-1, keepdims=True) + 1e-8
        goal_repr = goal_latents / goal_latents_norm * jnp.sqrt(repr_dim)
        goal_repr = jax.lax.stop_gradient(goal_repr)
        policy_obs = jnp.concatenate([state, goal_repr], axis=-1)
        logits = skill_actor.apply(skill_actor_params, policy_obs)
        if deterministic:
            action = parametric_action_distribution.mode(logits)
        else:
            action = parametric_action_distribution.sample(logits, key_sample)
        extras = {}
        return action, extras
    return policy

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    skill_actor_state: TrainState
    skill_critic1_state: TrainState
    skill_critic2_state: TrainState
    skill_value_state: TrainState
    value1_state: TrainState
    value2_state: TrainState
    # Target params (no optimizer state needed)
    value1_target_params: Any = None
    value2_target_params: Any = None
    skill_critic1_target_params: Any = None
    skill_critic2_target_params: Any = None

def _init_training_state(key, skill_actor, value1, value2, skill_critic1, skill_critic2, skill_value, state_dim, goal_dim, action_dim, repr_dim, actor_lr, critic_lr, num_local_devices_to_use):
    """
    Initializes the training state for a forward-backward representation learning model.
    """
    skill_actor_key, value1_key, value2_key, skill_critic1_key, skill_critic2_key, skill_value_key = jax.random.split(key, 6)
    
    # Actor
    skill_actor_params = skill_actor.init(skill_actor_key, jnp.ones([1, state_dim + repr_dim]))
    skill_actor_state = TrainState.create(apply_fn=skill_actor.apply, params=skill_actor_params, tx=optax.adam(learning_rate=actor_lr))

    # Critic and Value
    skill_critic1_params = skill_critic1.init(skill_critic1_key, jnp.ones([1, state_dim + action_dim + repr_dim]))
    skill_critic2_params = skill_critic2.init(skill_critic2_key, jnp.ones([1, state_dim + action_dim + repr_dim]))
    skill_value_params = skill_value.init(skill_value_key, jnp.ones([1, state_dim + repr_dim]))
    # MetricValueNet requires both observations and goals
    value1_params = value1.init(value1_key, jnp.ones([1, goal_dim]), jnp.ones([1, goal_dim]))
    value2_params = value2.init(value2_key, jnp.ones([1, goal_dim]), jnp.ones([1, goal_dim]))
    
    skill_critic1_state = TrainState.create(apply_fn=skill_critic1.apply, params=skill_critic1_params, tx=optax.adam(learning_rate=critic_lr))
    skill_critic2_state = TrainState.create(apply_fn=skill_critic2.apply, params=skill_critic2_params, tx=optax.adam(learning_rate=critic_lr))
    skill_value_state = TrainState.create(apply_fn=skill_value.apply, params=skill_value_params, tx=optax.adam(learning_rate=critic_lr))
    value1_state = TrainState.create(apply_fn=value1.apply, params=value1_params, tx=optax.adam(learning_rate=critic_lr))
    value2_state = TrainState.create(apply_fn=value2.apply, params=value2_params, tx=optax.adam(learning_rate=critic_lr))

    training_state = TrainingState(
        env_steps=jnp.zeros(()), 
        gradient_steps=jnp.zeros(()), 
        skill_actor_state=skill_actor_state,
        skill_critic1_state=skill_critic1_state,
        skill_critic2_state=skill_critic2_state,
        skill_value_state=skill_value_state,
        value1_state=value1_state,
        value2_state=value2_state,
        value1_target_params=value1_params,
        value2_target_params=value2_params,
        skill_critic1_target_params=skill_critic1_params,
        skill_critic2_target_params=skill_critic2_params,
    )
    
    training_state = jax.device_put_replicated(training_state, jax.local_devices()[:num_local_devices_to_use])
    return training_state


def value_head_loss(value_head_params, value_head, value1_target_params, value1_target_head, value2_target_params, value2_target_head, transitions, state_dim, discount=0.99, expectile=0.9):
    """Per-head value loss using target ensemble for advantages and per-head target for q."""
    # Extract observations, next_observations, value_goals, relabeled_rewards, relabeled_masks
    state_goal_portion = transitions.extras["state_goal_portion"]
    next_state_goal_portion = transitions.extras["next_state_goal_portion"]
    value_goals = transitions.extras["value_goals"]
    # Construct next_observations: next_state + goal portion (from value_goals or original goal)
    # Note: MetricValueNet only uses state portion via phi, but we keep full observation structure
    relabeled_rewards = transitions.extras["relabeled_rewards"]
    relabeled_masks = transitions.extras["relabeled_mask"]
    
    # Compute next_v using value network
    # Targets for next state
    next_v1_t = value1_target_head.apply(value1_target_params, next_state_goal_portion, value_goals[:, state_dim:], s_is_phi=False, g_is_phi=False)
    next_v2_t = value2_target_head.apply(value2_target_params, next_state_goal_portion, value_goals[:, state_dim:], s_is_phi=False, g_is_phi=False)
    next_v_min = jnp.minimum(next_v1_t, next_v2_t)
    
    # Compute Q using relabeled rewards and masks
    q = relabeled_rewards + discount * relabeled_masks * next_v_min
    
    # Compute v_t using target value network (for advantage computation)
    v1_t = value1_target_head.apply(value1_target_params, state_goal_portion, value_goals[:, state_dim:], s_is_phi=False, g_is_phi=False)
    v2_t = value2_target_head.apply(value2_target_params, state_goal_portion, value_goals[:, state_dim:], s_is_phi=False, g_is_phi=False)
    v_t = (v1_t + v2_t) / 2.0
    adv = q - v_t
    
    # Expectile loss: use advantage (from target) as weight, but difference (from current) for loss
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)
    
    # Head-specific q target and online v
    if value_head is value1_target_head:
        q_head = relabeled_rewards + discount * relabeled_masks * next_v1_t
    else:
        q_head = relabeled_rewards + discount * relabeled_masks * next_v2_t
    v_online = value_head.apply(value_head_params, state_goal_portion, value_goals[:, state_dim:], s_is_phi=False, g_is_phi=False)
    value_loss_total = expectile_loss(adv, q_head - v_online, expectile).mean()
    
    metrics = {
        'value_loss': value_loss_total,
        'v_mean': v_online.mean(),
        'v_max': v_online.max(),
        'v_min': v_online.min(),
    }
    return value_loss_total, metrics

def skill_actor_loss(skill_actor_params, training_state, skill_actor, skill_value, skill_critic1, skill_critic2, value1, parametric_action_distribution, transitions, state_dim, goal_dim, repr_dim, key, alpha=10.0):
    """Compute the HILP-style actor loss (AWR-style), using sampled random skills z.

    z is sampled randomly and treated as the conditioning variable.
    """
    observations = transitions.observation[:, :state_dim]
    if "skills" not in transitions.extras:
        raise KeyError("Expected transitions.extras['skills'] to be populated in update_step.")
    skills = transitions.extras["skills"]
    actions = transitions.action

    # Value and critic evaluations conditioned on skills.
    v = skill_value.apply(
        training_state.skill_value_state.params,
        jnp.concatenate([observations, skills], axis=-1),
    )
    q1 = skill_critic1.apply(
        training_state.skill_critic1_state.params,
        jnp.concatenate([observations, actions, skills], axis=-1),
    )
    q2 = skill_critic2.apply(
        training_state.skill_critic2_state.params,
        jnp.concatenate([observations, actions, skills], axis=-1),
    )
    # Shape safety: heads often output (B, 1); force (B,) to avoid (B,) vs (B,1) -> (B,B) broadcasting.
    v = jnp.reshape(v, (v.shape[0],))
    q1 = jnp.reshape(q1, (q1.shape[0],))
    q2 = jnp.reshape(q2, (q2.shape[0],))
    q = jnp.minimum(q1, q2)
    adv = q - v

    exp_a = jnp.exp(adv * alpha)
    exp_a = jnp.minimum(exp_a, 100.0)

    policy_inputs = jnp.concatenate([observations, skills], axis=-1)
    dist_params = skill_actor.apply(skill_actor_params, policy_inputs)
    log_prob = parametric_action_distribution.log_prob(dist_params, actions)
    actor_loss = -(exp_a * log_prob).mean()

    means, log_stds = jnp.split(dist_params, 2, axis=-1)
    mode = parametric_action_distribution.mode(dist_params)
    metrics = {
        'actor_loss': actor_loss,
        'actor_log_prob': log_prob.mean(),
        'adv': adv.mean(),
        'mse': jnp.mean((mode - actions) ** 2),
        'std': jnp.mean(jnp.exp(log_stds)),
    }
    return actor_loss, metrics

def skill_critic_loss(skill_critic_params, skill_value_params, value_params, skill_critic, skill_value, value1, transitions, state_dim, discount=0.99):
    """Compute the IQL critic loss for a single skill critic head (pretraining).

    TD target uses intrinsic skill_rewards = (phi(s')-phi(s))·z
    with z sampled randomly (provided in transitions.extras['skills']).
    """
    states = transitions.observation[:, :state_dim]
    actions = transitions.action
    next_states = transitions.extras["next_state"][:, :state_dim]
    if "skills" not in transitions.extras or "skill_rewards" not in transitions.extras:
        raise KeyError("Expected transitions.extras['skills'] and ['skill_rewards'] to be populated in update_step.")
    skills = transitions.extras["skills"]
    rewards = transitions.extras["skill_rewards"]
    masks = transitions.extras.get("mask", jnp.ones_like(rewards))
    # Shape safety: force (B,) for all scalar tensors to avoid (B,) vs (B,1) -> (B,B) broadcasting.
    rewards = jnp.reshape(rewards, (rewards.shape[0],))
    masks = jnp.reshape(masks, (masks.shape[0],))

    # Compute next_v using value network
    next_v = skill_value.apply(skill_value_params, jnp.concatenate([next_states, skills], axis=-1))
    next_v = jnp.reshape(next_v, (next_v.shape[0],))

    # Get q from critic head
    q1 = skill_critic.apply(skill_critic_params, jnp.concatenate([states, actions, skills], axis=-1))
    q1 = jnp.reshape(q1, (q1.shape[0],))

    # Compute target q using the provided discount constant
    q = rewards + discount * masks * next_v

    # Compute critic loss as mean squared error for both q1 and q2
    skill_critic_loss = ((q1 - q) ** 2).mean()

    metrics = {
        'skill_critic_loss': skill_critic_loss,
        'q_mean': q.mean(),
        'q_max': q.max(),
        'q_min': q.min(),
    }
    return skill_critic_loss, metrics

def skill_value_loss(skill_value_params, training_state, skill_value, skill_critic1, skill_critic2, value1, transitions, state_dim, expectile=0.9):
    """Compute the IQL value loss for skill_value (pretraining) using sampled random skills z."""
    # Unpack states, actions, goals
    states = transitions.observation[:, :state_dim]
    actions = transitions.action
    if "skills" not in transitions.extras:
        raise KeyError("Expected transitions.extras['skills'] to be populated in update_step.")
    skills = transitions.extras["skills"]
    # Compute Q-values from target skill critics (min)
    q1 = skill_critic1.apply(training_state.skill_critic1_target_params, jnp.concatenate([states, actions, skills], axis=-1))
    q2 = skill_critic2.apply(training_state.skill_critic2_target_params, jnp.concatenate([states, actions, skills], axis=-1))
    q1 = jnp.reshape(q1, (q1.shape[0],))
    q2 = jnp.reshape(q2, (q2.shape[0],))
    q = jnp.minimum(q1, q2)

    # Compute value estimates
    v = skill_value.apply(skill_value_params, jnp.concatenate([states, skills], axis=-1))
    v = jnp.reshape(v, (v.shape[0],))

    # Expectile loss (as in fb_repr.py)
    diff = q - v
    weight = jnp.where(diff >= 0, expectile, 1 - expectile)
    skill_value_loss = jnp.mean(weight * (diff ** 2))

    metrics = {
        'skill_value_loss': skill_value_loss,
        'v_mean': v.mean(),
        'v_max': v.max(),
        'v_min': v.min(),
    }
    return skill_value_loss, metrics


def actor_step_latent(env, env_state, skill_actor, parametric_action_distribution, skill_actor_params, latents, key, extra_fields=()):
    """
    Executes one step of an actor in the environment by selecting an action based on the
    policy, stepping the environment, and returning the updated state and transition data.

    Parameters
    ----------
    env : Env
        The environment in which the actor operates.
    env_state : State
        The current state of the environment.
    actor : brax.training.types.Policy
        The policy used to select the action.
    parametric_action_distribution : brax.training.distribution.ParametricDistribution
        A tanh normal distribution, used to map the actor's output to an action vector with elements between [-1, 1].
    actor_params : Any
        Parameters for the actor network.
    key : PRNGKey
        A random key for stochastic policy decisions.
    extra_fields : Sequence[str], optional
        A sequence of extra fields to be extracted from the environment state.

    Returns
    -------
    Tuple[State, Transition]
        A tuple containing the new state after taking the action and the transition data
        encompassing observation, action, reward, discount, and extra information.

    """
    policy_obs = jnp.concatenate([env_state.obs[:, :env.state_dim], latents], axis=1)
    action_mean_and_SD = skill_actor.apply(skill_actor_params, policy_obs)
    action = parametric_action_distribution.sample(action_mean_and_SD, key)
    nstate = env.step(env_state, action)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, Transition(
        observation=env_state.obs,  # keep raw observation [state, goal]
        action=action,
        reward=nstate.reward,
        discount=1 - nstate.done,
        extras={"policy_extras": {}, "state_extras": state_extras},
    )
    
def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def sample_latents(value_net, value_params, batch, latent_dim, key):
    """Sample normalized latents and compute intrinsic rewards ala HILP."""
    states_goal_portion = batch["extras"]["state_goal_portion"]
    next_states_goal_portion = batch["extras"]["next_state_goal_portion"]
    dtype = batch["actions"].dtype
    batch_size = states_goal_portion.shape[0]

    latents = jax.random.normal(key, shape=(batch_size, latent_dim), dtype=dtype)
    latents = latents / (jnp.linalg.norm(latents, axis=1, keepdims=True) + 1e-8) * jnp.sqrt(latent_dim)

    _, phis, next_phis = value_net.apply(value_params, states_goal_portion, next_states_goal_portion, info=True)
    rewards = jnp.sum((next_phis - phis) * latents, axis=1)

    return latents, rewards


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
    actor_freq: int = 1,
):
    """
    Trains a forward-backward representation learning agent.
    """

    # Reproducibility preparation for (optional) multi-GPU training
    process_id = jax.process_index()
    num_local_devices_to_use = jax.local_device_count()
    device_count = num_local_devices_to_use * jax.process_count()
    logging.info(
        "local_device_count: %s; total_device_count: %s",
        num_local_devices_to_use,
        device_count,
    )

    # Sanity checks
    if min_replay_size >= num_timesteps:
        raise ValueError("No training will happen because min_replay_size >= num_timesteps")

    if ((episode_length - 1) * num_envs) % batch_size != 0:
        raise ValueError("(episode_length - 1) * num_envs must be divisible by batch_size")

    if max_replay_size is None:
        max_replay_size = num_timesteps

    # The number of environment steps executed for every `actor_step()` call.
    env_steps_per_actor_step = action_repeat * num_envs * unroll_length
    num_prefill_actor_steps = min_replay_size // unroll_length + 1
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert num_timesteps - min_replay_size >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps) // (num_evals_after_init * env_steps_per_actor_step)
    )

    assert num_envs % device_count == 0
    env = environment
    wrap_for_training = envs.training.wrap

    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    env = TrajectoryIdWrapper(env)
    env = wrap_for_training(env, episode_length=episode_length, action_repeat=action_repeat)
    unwrapped_env = environment
    env_train_context = wrap_for_training(environment, episode_length=episode_length, action_repeat=action_repeat)

    obs_size = env.observation_size
    action_size = env.action_size
    state_dim = env.state_dim
    goal_dim = obs_size - state_dim
    goal_indices = env.goal_indices
    
    dummy_obs = jnp.zeros((state_dim + goal_dim,))
    dummy_action = jnp.zeros((action_size,))
    dummy_extras = {"state_extras": {"truncation": 0.0, "traj_id": 0.0}, "policy_extras": {}}
    dummy_transition = Transition(observation=dummy_obs, action=dummy_action, reward=0.0, discount=0.0, extras=dummy_extras)
    
    replay_buffer = TrajectoryUniformSamplingQueue(
        max_replay_size=max_replay_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size // device_count,
        num_envs=num_envs,
        episode_length=episode_length,
    )
    replay_buffer = jit_wrap(replay_buffer)
    
    # Network functions
    block_size = 2
    num_blocks = max(1, n_hidden // block_size)
    skill_actor = Net(action_size * 2, h_dim, num_blocks, block_size, use_ln)
    skill_critic1 = Net(1, h_dim, num_blocks, block_size, use_ln)  # Critic head 1
    skill_critic2 = Net(1, h_dim, num_blocks, block_size, use_ln)  # Critic head 2
    skill_value = Net(1, h_dim, num_blocks, block_size, use_ln)   # Outputs a single V-value
    # Two value heads (each computes v and phi internally)
    value1 = MetricValueNet(repr_dim, h_dim, num_blocks, block_size, use_ln)
    value2 = MetricValueNet(repr_dim, h_dim, num_blocks, block_size, use_ln)
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)

    # Initialize training state
    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)    
    training_state = _init_training_state(global_key, skill_actor, value1, value2, skill_critic1, skill_critic2, skill_value, state_dim, len(env.goal_indices), env.action_size, repr_dim, policy_lr, repr_lr, num_local_devices_to_use)
    del global_key
    
    # Update functions
    
    skill_actor_update = gradients.gradient_update_fn(skill_actor_loss, training_state.skill_actor_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    skill_critic1_update = gradients.gradient_update_fn(skill_critic_loss, training_state.skill_critic1_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    skill_critic2_update = gradients.gradient_update_fn(skill_critic_loss, training_state.skill_critic2_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    skill_value_update = gradients.gradient_update_fn(skill_value_loss, training_state.skill_value_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    value1_update = gradients.gradient_update_fn(value_head_loss, training_state.value1_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    value2_update = gradients.gradient_update_fn(value_head_loss, training_state.value2_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    
    def _soft_update(target_params, online_params, tau):
        return jax.tree_util.tree_map(lambda tp, p: tau * p + (1.0 - tau) * tp, target_params, online_params)

    def update_step(carry, transitions):
        training_state, key = carry
        key, key_fb, key_actor, key_critic, key_value = jax.random.split(key, 5)

        # ---------------------------------------------------------------------
        # HILP pretraining (skills): sample random z and intrinsic skill rewards
        #   skill_rewards = (phi(s') - phi(s)) · z
        # ---------------------------------------------------------------------
        skill_batch = {"extras": transitions.extras, "actions": transitions.action}
        skills, skill_rewards = sample_latents(
            value_net=value1,
            value_params=training_state.value1_state.params,
            batch=skill_batch,
            latent_dim=repr_dim,
            key=key_fb,
        )
        transitions = transitions._replace(
            extras={**transitions.extras, "skills": skills, "skill_rewards": skill_rewards}
        )
        
        # Update value heads with target ensemble
        (value1_loss_val, value1_metrics), value1_params, value1_optimizer_state = value1_update(
            training_state.value1_state.params,
            value1,  # online head
            training_state.value1_target_params, value1,  # target head 1
            training_state.value2_target_params, value2,  # target head 2
            transitions,
            env.state_dim,
            0.99,
            0.9,
            optimizer_state=training_state.value1_state.opt_state
        )
        (value2_loss_val, value2_metrics), value2_params, value2_optimizer_state = value2_update(
            training_state.value2_state.params,
            value2,  # online head
            training_state.value1_target_params, value1,  # target head 1
            training_state.value2_target_params, value2,  # target head 2
            transitions,
            env.state_dim,
            0.99,
            0.9,
            optimizer_state=training_state.value2_state.opt_state
        )
        
        # Update skill value function
        (skill_value_loss, skill_value_metrics), skill_value_params, skill_value_optimizer_state = skill_value_update(
            training_state.skill_value_state.params,
            training_state,
            skill_value,
            skill_critic1,
            skill_critic2,
            value1,
            transitions,
            env.state_dim,
            0.9,
            optimizer_state=training_state.skill_value_state.opt_state
        )
        
        # Update skill critics independently
        (skill_critic1_loss, skill_critic1_metrics), skill_critic1_params, skill_critic1_optimizer_state = skill_critic1_update(
            training_state.skill_critic1_state.params,
            training_state.skill_value_state.params,
            training_state.value1_state.params,
            skill_critic1,
            skill_value,
            value1,
            transitions,
            env.state_dim,
            0.99,
            optimizer_state=training_state.skill_critic1_state.opt_state
        )
        (skill_critic2_loss, skill_critic2_metrics), skill_critic2_params, skill_critic2_optimizer_state = skill_critic2_update(
            training_state.skill_critic2_state.params,
            training_state.skill_value_state.params,
            training_state.value1_state.params,
            skill_critic2,
            skill_value,
            value1,
            transitions,
            env.state_dim,
            0.99,
            optimizer_state=training_state.skill_critic2_state.opt_state
        )
        
        # Update actor (gate by actor_freq)
        next_step = training_state.gradient_steps + 1
        (computed_actor_loss, skill_actor_metrics), new_actor_params, new_actor_opt_state = skill_actor_update(
            training_state.skill_actor_state.params,
            training_state,
            skill_actor,
            skill_value,
            skill_critic1,
            skill_critic2,
            value1,
            parametric_action_distribution,
            transitions,
            env.state_dim,
            len(env.goal_indices),
            repr_dim,
            key_actor,
            optimizer_state=training_state.skill_actor_state.opt_state
        )
        do_actor_update = (next_step % actor_freq) == 0
        do_actor_update = jnp.asarray(do_actor_update)
        tree_select = lambda a, b: jax.tree_util.tree_map(lambda x, y: jax.lax.select(do_actor_update, x, y), a, b)
        skill_actor_params = tree_select(new_actor_params, training_state.skill_actor_state.params)
        skill_actor_optimizer_state = tree_select(new_actor_opt_state, training_state.skill_actor_state.opt_state)
        skill_actor_loss = jax.lax.select(do_actor_update, computed_actor_loss, jnp.asarray(0.0))
        skill_actor_metrics = {**skill_actor_metrics, "actor_update_applied": jnp.asarray(do_actor_update, dtype=jnp.float32)}
    

        metrics = {
            'skill_actor_loss': skill_actor_loss,
            'skill_critic1_loss': skill_critic1_loss,
            'skill_critic2_loss': skill_critic2_loss,
            'skill_value_loss': skill_value_loss,
            'value1_loss': value1_loss_val,
            'value2_loss': value2_loss_val,
        }
        metrics.update(skill_actor_metrics)
        metrics.update({'critic1/'+k: v for k, v in skill_critic1_metrics.items()})
        metrics.update({'critic2/'+k: v for k, v in skill_critic2_metrics.items()})
        metrics.update({'value1/'+k: v for k, v in value1_metrics.items()})
        metrics.update({'value2/'+k: v for k, v in value2_metrics.items()})
        metrics.update(skill_value_metrics)

        # Soft-update targets
        new_value1_target_params = _soft_update(training_state.value1_target_params, value1_params, tau)
        new_value2_target_params = _soft_update(training_state.value2_target_params, value2_params, tau)
        new_skill_critic1_target_params = _soft_update(training_state.skill_critic1_target_params, skill_critic1_params, tau)
        new_skill_critic2_target_params = _soft_update(training_state.skill_critic2_target_params, skill_critic2_params, tau)

        new_training_state = TrainingState(
            env_steps=training_state.env_steps,
            gradient_steps=training_state.gradient_steps + 1,
            skill_actor_state=training_state.skill_actor_state.replace(params=skill_actor_params, opt_state=skill_actor_optimizer_state),
            value1_state=training_state.value1_state.replace(params=value1_params, opt_state=value1_optimizer_state),
            value2_state=training_state.value2_state.replace(params=value2_params, opt_state=value2_optimizer_state),
            skill_critic1_state=training_state.skill_critic1_state.replace(params=skill_critic1_params, opt_state=skill_critic1_optimizer_state),
            skill_critic2_state=training_state.skill_critic2_state.replace(params=skill_critic2_params, opt_state=skill_critic2_optimizer_state),
            skill_value_state=training_state.skill_value_state.replace(params=skill_value_params, opt_state=skill_value_optimizer_state),
            value1_target_params=new_value1_target_params,
            value2_target_params=new_value2_target_params,
            skill_critic1_target_params=new_skill_critic1_target_params,
            skill_critic2_target_params=new_skill_critic2_target_params,
        )
        
        return (new_training_state, key), metrics

    def get_experience(skill_actor_params, value_params, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            # For experience collection we keep conditioning on goal-derived latents for now.
            # (Pretraining skills z are sampled per-minibatch during updates.)
            states = env_state.obs[:, :state_dim]
            goals = env_state.obs[:, state_dim:]
            states_goal_portion = states[:, goal_indices]
            _, phi_states, phi_goals = value1.apply(value_params, states_goal_portion, goals, info=True)
            step_latents = phi_goals - phi_states
            step_latents = step_latents / (jnp.linalg.norm(step_latents, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(repr_dim)
            step_latents = jax.lax.stop_gradient(step_latents)
            env_state, transition = actor_step_latent(env, env_state, skill_actor, parametric_action_distribution, skill_actor_params, step_latents, current_key, extra_fields=("truncation", "traj_id"))
            return (env_state, next_key), transition
        
       # Split the key to create a batch of keys matching env_state.obs.shape[0]
        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=episode_length)
        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def training_step(training_state, env_state, buffer_state, key):
        # Collect experience
        experience_key, training_key = jax.random.split(key, 2)
        env_state, buffer_state = get_experience(training_state.skill_actor_state.params, training_state.value1_state.params, env_state, buffer_state, experience_key)
        training_state = training_state.replace(env_steps=training_state.env_steps + env_steps_per_actor_step)
        
        # Train
        training_state, buffer_state, metrics = train_steps(training_state, buffer_state, training_key)
        return training_state, env_state, buffer_state, metrics

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        def f(carry, unused):
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(training_state.skill_actor_state.params, training_state.value1_state.params, env_state, buffer_state, key)
            new_training_state = training_state.replace(env_steps=training_state.env_steps + env_steps_per_actor_step)
            return (new_training_state, env_state, buffer_state, new_key), ()
        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=num_prefill_actor_steps)[0]
    
    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

    def train_steps(training_state, buffer_state, key):
        # Sample from buffer
        experience_key, training_key, sampling_key = jax.random.split(key, 3)
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        
        # Process transitions using flatten_crl_fn (vmap)
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        vmap_flatten_crl_fn = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, None, 0, 0))
        transitions = vmap_flatten_crl_fn(config, env, transitions, batch_keys)
        
        # Shuffle and reshape transitions
        transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions)
        permutation = jax.random.permutation(experience_key, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]), transitions)
        
        # Train
        (training_state, _), metrics = jax.lax.scan(update_step, (training_state, training_key), transitions)
        return training_state, buffer_state, metrics

    def training_epoch(training_state, env_state, buffer_state, key):
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key = jax.random.split(k)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            return (ts, es, bs, new_key), metrics
        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=num_training_steps_per_epoch)
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    def training_epoch_with_timing(training_state, env_state, buffer_state, key):
        nonlocal training_walltime
        t = time.time()
        (training_state, env_state, buffer_state, metrics) = training_epoch(training_state, env_state, buffer_state, key)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return (training_state, env_state, buffer_state, metrics)

    # Initialization and setup
    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)
    env_keys = jax.random.split(env_key, num_envs // jax.process_count())
    env_keys = jnp.reshape(env_keys, (num_local_devices_to_use, -1) + env_keys.shape[1:])
    env_state = jax.pmap(env.reset)(env_keys)

    # Replay buffer init and prefill
    buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, num_local_devices_to_use))
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, num_local_devices_to_use)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(training_state, env_state, buffer_state, prefill_keys)
    replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    assert replay_size >= min_replay_size
    training_walltime = time.time() - t

    # Eval init
    if not eval_env:
        eval_env = environment
    eval_env = TrajectoryIdWrapper(eval_env)
    eval_env = wrap_for_training(eval_env, episode_length=episode_length, action_repeat=action_repeat)
    global make_policy
    make_policy = functools.partial(
        make_policy,
        skill_actor,
        parametric_action_distribution,
        value1,
        goal_indices=env.goal_indices,
        state_dim=env.state_dim,
        repr_dim=repr_dim,
    )
    evaluator = CrlEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        # We pass in the actor and backward_repr params to the evaluator
        eval_params = _unpmap((training_state.skill_actor_state.params, training_state.value1_state.params))
        metrics = evaluator.run_evaluation(eval_params, training_metrics={})
        logging.info(metrics)
        progress_fn(0, metrics, make_policy, eval_params, unwrapped_env)

    # Collect/train/eval loop
    current_step = 0
    for eval_epoch_num in range(num_evals_after_init):
        logging.info("step %s", current_step)

        # Collect data and train
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, num_local_devices_to_use)
        (training_state, env_state, buffer_state, training_metrics) = training_epoch_with_timing(training_state, env_state, buffer_state, epoch_keys)
        current_step = int(_unpmap(training_state.env_steps))

        # Logging and evals
        if process_id == 0:
            ## Save policy and representation params
            if checkpoint_logdir:
                params = _unpmap((
                    training_state.skill_actor_state.params,
                    training_state.skill_value_state.params,
                    training_state.value1_state.params,
                    training_state.value2_state.params,
                    training_state.skill_critic1_state.params,
                    training_state.skill_critic2_state.params,
                ))
                path = f"{checkpoint_logdir}/step_{current_step}.pkl"
                # Log all params
                logging.info(f"Saving checkpoint at {path} with actor, fb_repr, and target params.")
                brax.io.model.save_params(path, params)
            ## Run evals
            eval_params = _unpmap((training_state.skill_actor_state.params, training_state.value1_state.params))
            metrics = evaluator.run_evaluation(eval_params, training_metrics)
            logging.info(metrics)
            progress_fn(current_step, metrics, make_policy, eval_params, unwrapped_env)

    # Final validity checks
    total_steps = current_step
    logging.info("total steps: %s", total_steps)
    assert total_steps >= num_timesteps

    pmap.assert_is_replicated(training_state)
    pmap.synchronize_hosts()
    
    params = _unpmap((
        training_state.skill_actor_state.params,
        training_state.skill_value_state.params,
        training_state.value1_state.params,
        training_state.value2_state.params,
        training_state.skill_critic1_state.params,
        training_state.skill_critic2_state.params,
    ))
    # Log all params at the end as well
    logging.info("Returning actor, fb_repr, and target params.")
    return (make_policy, params, metrics)

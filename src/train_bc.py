import functools
import time
from typing import Callable, Optional, NamedTuple

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

# The brax version of this does not take in the actor and action_distribution arguments; before we pass it to brax evaluator or return it from train(), we do a partial application.
def make_policy(actor, parametric_action_distribution, params, deterministic=False):
    def policy(obs, key_sample):
        print("obs shape", obs.shape)
        logits = actor.apply(params, obs)
        print("LOGITS", logits.shape)
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
    alpha_state: TrainState
    context_state: TrainState

def _init_training_state(key, actor, context_encoder, var_post, state_dim, goal_dim, action_dim, episode_length, actor_lr, alpha_lr, num_local_devices_to_use):
    """
    Initializes the training state for a contrastive reinforcement learning model. This function sets up the initial states for various components including the policy
    network, CRL networks, and optimizers. All parameters are initialized and replicated across the specified
    number of local devices.

    Args:
        key (PRNGKey): A pseudorandom number generator key used for initializing various network parameters.
        actor (function): The flax function for the actor network.
        sa_encoder (function): The flax function for the state-action encoder network.
        g_encoder (function): The flax function for the goal encoder network.
        state_dim (int): The dimension of the observations from the environment, not including goals.
        goal_dim (int): The dimension of the goals in the environment.
        action_dim (int): The dimension of the action that the actor should pass into the environment in env.step.
        actor_lr (float): The learning rate for the actor network.
        critic_lr (float): The learning rate for the state-action and goal encoder networks.
        alpha_lr (float): The learning rate for the entropy coefficient (alpha).
        num_local_devices_to_use (int): The number of local devices to utilize for training.

    Returns:
        TrainingState: An initialized TrainingState object that contains the initial states of the
        policy network parameters, CRL critic parameters, their respective optimizer states, alpha parameter,
        and normalization parameters.
    """
    actor_key, context_key = jax.random.split(key, 2)
    # print(state_dim, goal_dim, action_dim)
    # Actor and entropy coefficient
    actor_params = actor.init(actor_key, jnp.ones([1, state_dim + goal_dim]))
    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(learning_rate=actor_lr))
    log_alpha = {"log_alpha": jnp.array(0.0)}
    alpha_state = TrainState.create(apply_fn=None, params=log_alpha, tx=optax.adam(learning_rate=alpha_lr))

    if var_post == "meanfield":
        dummy_input = jnp.ones([1, (state_dim + action_dim)])
    else:
        dummy_input = jnp.ones([1, (episode_length - 1) * (state_dim + action_dim)])
    context_encoder_params = context_encoder.init(context_key, dummy_input)
    context_state = TrainState.create(apply_fn=None, params=context_encoder_params, tx=optax.adam(learning_rate=actor_lr))
    training_state = TrainingState(env_steps=jnp.zeros(()), gradient_steps=jnp.zeros(()), 
                                   actor_state=actor_state,  alpha_state=alpha_state,
                                   context_state=context_state)
    training_state = jax.device_put_replicated(training_state, jax.local_devices()[:num_local_devices_to_use])
    return training_state

def compute_energy(energy_fn, sa_repr, g_repr):
    if energy_fn == "l2":
        logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
    elif energy_fn == "l1":
        logits = -jnp.sum(jnp.abs(sa_repr[:, None, :] - g_repr[None, :, :]), axis=-1)
    elif energy_fn == "dot":
        logits = jnp.einsum("ik,jk->ij", sa_repr, g_repr)
    else:
        raise ValueError(f"Unknown energy function: {energy_fn}")
    return logits
        
def compute_actor_energy(energy_fn, sa_repr, g_repr):
    if energy_fn == "l2":
        q = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
    elif energy_fn == "l1":
        q = -jnp.sum(jnp.abs(sa_repr - g_repr), axis=-1)
    elif energy_fn == "dot":
        q = jnp.einsum("ik,ik->i", sa_repr, g_repr)
    else:
        raise ValueError(f"Unknown energy function: {energy_fn}")
    return q
        
# Helper for compute_loss
def log_softmax(logits, axis, resubs):
    if not resubs:
        I = jnp.eye(logits.shape[0])
        big = 100
        eps = 1e-6
        return logits, -jax.nn.logsumexp(logits - big * I + eps, axis=axis, keepdims=True)
    else:
        return logits, -jax.nn.logsumexp(logits, axis=axis, keepdims=True)
    
def compute_loss(contrastive_loss_fn, logits, resubs):
    if contrastive_loss_fn == "symmetric_infonce":
        l_align1, l_unify1 = log_softmax(logits, axis=1, resubs=resubs)
        l_align2, l_unify2 = log_softmax(logits, axis=0, resubs=resubs)
        l_align = l_align1 + l_align2
        l_unif = l_unify1 + l_unify2
        loss = -jnp.mean(jnp.diag(l_align1 + l_unify1) + jnp.diag(l_align2 + l_unify2))
    elif contrastive_loss_fn == "infonce":
        # From "Improved Deep Metric Learning with Multi-class N-pair Loss Objective"
        # https://dl.acm.org/doi/10.5555/3157096.3157304
        l_align, l_unif = log_softmax(logits, axis=1, resubs=resubs)
        loss = -jnp.mean(jnp.diag(l_align + l_unif))
    elif contrastive_loss_fn == "infonce_backward":
        l_align, l_unif = log_softmax(logits, axis=0, resubs=resubs)
        loss = -jnp.mean(jnp.diag(l_align + l_unif))
    elif contrastive_loss_fn == "flatnce":
        # From "Simpler, Faster, Stronger: Breaking The log-K Curse
        # On Contrastive Learners With FlatNCE" https://arxiv.org/pdf/2107.01152
        logits_flat = logits - jnp.diag(logits)[:, None]
        clogits = jax.nn.logsumexp(logits_flat, axis=1)
        l_align = clogits
        l_unif = jnp.sum(logits_flat, axis=-1)
        loss = jnp.exp(clogits - jax.lax.stop_gradient(clogits)).mean()
    elif contrastive_loss_fn == "dpo":
        # Based on "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
        # https://arxiv.org/pdf/2305.18290
        # It aims to drive positive and negative logits further away from each other
        positive = jnp.diag(logits)
        diffs = positive[:, None] - logits
        loss = -jnp.mean(jax.nn.log_sigmoid(diffs))
        l_align = 0
        l_unif = 0
    elif contrastive_loss_fn == "fb":
        # This is a Monte Carlo version of the loss from "Does Zero-Shot Reinforcement Learning Exist?"
        # https://arxiv.org/abs/2209.14935
        batch_size = logits.shape[0]
        I = jnp.eye(batch_size)
        l_align = -jnp.diag(logits)  # shape = (batch_size,)
        l_unif = 0.5 * jnp.sum(logits**2 * (1 - I) / (batch_size - 1), axis=-1)  # shape = (batch_size,)
        loss = (l_align + l_unif).mean()  # shape = ()
    else:
        raise ValueError(f"Unknown contrastive loss function: {contrastive_loss_fn}")
    return loss, l_align, l_unif

def compute_metrics(logits, sa_repr, g_repr, l2_loss, l_align, l_unif):
    I = jnp.eye(logits.shape[0])
    correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
    logits_pos = jnp.sum(logits * I) / jnp.sum(I)
    logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
    if len(logits.shape) == 3:
        logsumexp = jax.nn.logsumexp(logits[:, :, 0], axis=1) ** 2
    else:
        logsumexp = jax.nn.logsumexp(logits, axis=1) ** 2

    sa_repr_l2 = jnp.sqrt(jnp.sum(sa_repr**2, axis=-1))
    g_repr_l2 = jnp.sqrt(jnp.sum(g_repr**2, axis=-1))

    l_align_log = -jnp.mean(jnp.diag(l_align))
    l_unif_log = -jnp.mean(l_unif)

    metrics = {
        "binary_accuracy": jnp.mean((logits > 0) == I),
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "sa_repr_mean": jnp.mean(sa_repr_l2),
        "g_repr_mean": jnp.mean(g_repr_l2),
        "sa_repr_std": jnp.std(sa_repr_l2),
        "g_repr_std": jnp.std(g_repr_l2),
        "logsumexp": logsumexp.mean(),
        "l2_penalty": l2_loss,
        "l_align": l_align_log,
        "l_unif": l_unif_log,
    }
    return metrics

def alpha_loss(alpha_params, actor, parametric_action_distribution, training_state, transitions, action_size, key):
    """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
    action_mean_and_SD = actor.apply(training_state.actor_state.params, transitions.observation)
    action = parametric_action_distribution.sample_no_postprocessing(action_mean_and_SD, key)
    log_prob = parametric_action_distribution.log_prob(action_mean_and_SD, action)

    alpha = jnp.exp(alpha_params["log_alpha"])
    target_entropy = -0.5 * action_size

    alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
    return jnp.mean(alpha_loss)

    
def context_loss(
    context_encoder_params, context_encoder, transitions, state_dim, goal_indices, 
    latent_dim=5, beta=0.01, key=None
):
    """
    Computes mutual information loss to ensure the context encoder captures meaningful task information.
    
    Args:
        context_encoder_params: Parameters for the context encoder
        context_encoder: Context encoder network (similar structure to sa_encoder/g_encoder)
        transitions: Transitions from the replay buffer
        state_dim: Dimension of the state space
        goal_indices: Indices that define the goal
        latent_dim: Dimension of the latent space
        beta: Coefficient for the information loss
        key: JAX random key
    
    Returns:
        loss: The mutual information loss
        metrics: Dictionary of metrics related to the context encoding
    """
    # Split the random key
    key, encoder_key, sampling_key = jax.random.split(key, 3)
    
    print("transitions", transitions.observation.shape, transitions.reward.shape)
    # Extract states, actions, and goals from transitions
    states = transitions.observation[:, :, :state_dim]
    goals = transitions.observation[:, :, state_dim:]
    actions = transitions.action
    
    # Get the targets from the extras
    targets = goals[:, -1, :]
    print("targets", targets.shape)
    
    # Concatenate state-action to form trajectory representation
    trajectories = jnp.reshape(jnp.concatenate([states, actions], axis=-1), (-1, states.shape[-2] * (states.shape[-1] + actions.shape[-1])))
    
    print("trajectories", trajectories.shape)
    # Context encoder outputs a vector of size 2*latent_dim (first half is mean, second half is log_std)
    context_output = context_encoder.apply(context_encoder_params, trajectories)
    print("context output shape", context_output.shape)
    
    # Split the output into mean and log_std
    context_mean, context_log_std = jnp.split(context_output, 2, axis=-1)
    
    # Compute standard deviation from log_std, adding small epsilon for numerical stability
    epsilon = 1e-8
    std = jnp.exp(context_log_std) + epsilon
    
    # Compute log likelihood of targets under Gaussian distribution with predicted mean and std
    # Log likelihood = -0.5 * (log(2π) + log(σ²) + (x-μ)²/σ²)
    log_2pi = jnp.log(2 * jnp.pi)
    log_var = 2 * context_log_std  # log(σ²) = 2*log(σ)
    squared_mahalanobis = jnp.square(targets - context_mean) / jnp.square(std)
    
    log_likelihood = -0.5 * jnp.sum(
        log_2pi + log_var + squared_mahalanobis,
        axis=-1
    )  # Sum over dimensions, shape: [batch_size]
    
    # Maximize likelihood by minimizing negative log likelihood
    info_loss = -jnp.mean(log_likelihood)  # Average over batch
    
    # Collect metrics for monitoring
    metrics = {
        "context_info_loss": info_loss,
        "context_mean_norm": jnp.mean(jnp.sqrt(jnp.sum(jnp.square(context_mean), axis=-1))),
        "context_std_mean": jnp.mean(std),
    }
    
    return info_loss, metrics

def context_loss_meanfield(
    context_encoder_params, context_encoder, transitions, state_dim, goal_indices, 
    latent_dim=5, beta=0.01, key=None
):
    """
    Computes mutual information loss to ensure the context encoder captures meaningful task information.
    This version assumes the context encoder takes in state-action pairs rather than full trajectories.
    
    Args:
        context_encoder_params: Parameters for the context encoder
        context_encoder: Context encoder network (similar structure to sa_encoder/g_encoder)
        transitions: Transitions from the replay buffer
        state_dim: Dimension of the state space
        goal_indices: Indices that define the goal
        latent_dim: Dimension of the latent space
        beta: Coefficient for the information loss
        key: JAX random key
    
    Returns:
        loss: The mutual information loss
        metrics: Dictionary of metrics related to the context encoding
    """
    # Split the random key
    key, encoder_key, sampling_key = jax.random.split(key, 3)
    
    print("transitions", transitions.observation.shape, transitions.reward.shape)
    # Extract states, actions, and goals from transitions
    states = transitions.observation[:, :, :state_dim]
    goals = transitions.observation[:, :, state_dim:]
    actions = transitions.action
    
    # Get the targets from the extras
    targets = goals[:, -1, :]
    print("targets", targets.shape)
    # Create state-action pairs
    sa_pairs = jnp.concatenate([states, actions], axis=-1)
    batch_size, seq_len, sa_dim = sa_pairs.shape
    
    # Reshape to process each state-action pair individually
    sa_pairs_flat = sa_pairs.reshape(-1, sa_dim)  # Shape: [batch_size * seq_len, sa_dim]
    
    print("sa_pairs_flat", sa_pairs_flat.shape)
    # Context encoder outputs a vector of size 2*latent_dim for each state-action pair
    # (first half is mean, second half is log_std)
    context_output_flat = context_encoder.apply(context_encoder_params, sa_pairs_flat)
    print("context output shape", context_output_flat.shape)
    
    # Reshape back to batch structure
    context_output = context_output_flat.reshape(batch_size, seq_len, -1)
    print("context output shape", context_output.shape)
    
    # Split the output into mean and log_std for each state-action pair
    context_mean, context_log_std = jnp.split(context_output, 2, axis=-1)
    print("context mean shape", context_mean.shape)
    print("context log_std shape", context_log_std.shape)
    # Compute standard deviation from log_std, adding small epsilon for numerical stability
    epsilon = 1e-8
    std = jnp.exp(context_log_std) + epsilon
    
    # Compute log likelihood of targets under Gaussian distribution with predicted mean and std
    # for each state-action pair
    log_2pi = jnp.log(2 * jnp.pi)
    log_var = 2 * context_log_std  # log(σ²) = 2*log(σ)
    
    # Expand targets to match the shape of context_mean for broadcasting
    # targets shape: [batch_size, latent_dim]
    # context_mean shape: [batch_size, seq_len, latent_dim]
    targets_expanded = jnp.expand_dims(targets, axis=1)  # Shape: [batch_size, 1, latent_dim]
    
    squared_mahalanobis = jnp.square(targets_expanded - context_mean) / jnp.square(std)
    
    # Compute log likelihood for each state-action pair
    # Shape: [batch_size, seq_len]
    pair_log_likelihood = -0.5 * jnp.sum(
        log_2pi + log_var + squared_mahalanobis,
        axis=-1
    )
    print("pair_log_likelihood shape", pair_log_likelihood.shape)
    
    # Sum log likelihoods across the sequence to get the total log likelihood
    # This is equivalent to computing the product of probabilities
    # Shape: [batch_size]
    log_likelihood = jnp.sum(pair_log_likelihood, axis=1)
    print("log_likelihood shape", log_likelihood.shape)
    # Maximize likelihood by minimizing negative log likelihood
    info_loss = -jnp.mean(log_likelihood)  # Average over batch
    
    # Collect metrics for monitoring
    metrics = {
        "context_info_loss": info_loss,
        "context_mean_norm": jnp.mean(jnp.sqrt(jnp.sum(jnp.square(context_mean), axis=-1))),
        "context_std_mean": jnp.mean(std),
    }
    
    return info_loss, metrics


def actor_loss(actor_params, training_state, actor, parametric_action_distribution, alpha, transitions, config, state_dim, goal_indices, energy_fn_name, key):
    sample_key, entropy_key, goal_key = jax.random.split(key, 3)

    # Get state and goal
    state = transitions.observation[:, :state_dim]
    true_action = transitions.action
    goal = transitions.extras["target"]
    sg = jnp.concatenate([state, goal], axis=1)

    # Compute action with policy, given state and goal
    action_mean_and_SD = actor.apply(actor_params, sg)
    action = parametric_action_distribution.sample_no_postprocessing(action_mean_and_SD, sample_key)
    log_prob = parametric_action_distribution.log_prob(action_mean_and_SD, action)
    entropy = parametric_action_distribution.entropy(action_mean_and_SD, entropy_key)
    action = parametric_action_distribution.postprocess(action)
    
    # energy_fn_name = "nll"
    # Compute energy and loss
    if energy_fn_name == "nll":
        print("Using NLL loss")
        # Calculate log probability of true action under current policy
        q = parametric_action_distribution.log_prob(action_mean_and_SD, true_action)
    else: # "l2"
        print("Using L2 loss for continuous action spaces")
        print(action.shape, true_action.shape)
        q = -jnp.sum((action - true_action) ** 2, axis=-1)
        
    actor_loss = -jnp.mean(q)
    print("actor_loss", actor_loss.shape)
    # Modify loss (actor entropy)
    if not config.disable_entropy_actor:
        actor_loss += alpha * log_prob
        
    print("actor_loss", actor_loss.shape)
    
    # Compute metrics
    metrics = {"entropy": entropy.mean()}
    return jnp.mean(actor_loss), metrics


def actor_step(env, env_state, actor, parametric_action_distribution, actor_params, key, extra_fields=()):
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
    action_mean_and_SD = actor.apply(actor_params, env_state.obs)
    action = parametric_action_distribution.sample(action_mean_and_SD, key)
    nstate = env.step(env_state, action)
    # print(f"state.pipeline_state shape: {jax.tree_map(lambda x: x.shape, env_state.pipeline_state)}")
    # print(f"action shape: {jax.tree_map(lambda x: x.shape, action)}")
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, Transition(
        observation=env_state.obs,
        action=action,
        reward=nstate.reward,
        discount=1 - nstate.done,
        extras={"policy_extras": {}, "state_extras": state_extras},
    )

def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)

def train(
    environment: envs.Env,
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    policy_lr: float = 1e-4,
    alpha_lr: float = 1e-4,
    critic_lr: float = 1e-4,
    seed: int = 0,
    batch_size: int = 256,
    contrastive_loss_fn: str = "binary",
    energy_fn: str = "l2",
    logsumexp_penalty: float = 0.0,
    l2_penalty: float = 0.0,
    resubs: bool = True,
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
    visualization_interval: int = 5,
    var_post: str = "meanfield",
    saved_ckpt_path: Optional[str] = None,
):
    """
    Trains a contrastive reinforcement learning agent using the specified environment and parameters.

    This function initializes and manages the training process, including the setup of
    environments, networks, optimizers, replay buffers, and loss functions. It also
    handles the distribution of computation across multiple devices if available and
    configures the training loop to evaluate the agent's performance periodically.

    Parameters:
        environment: envs.Env
            The environment in which the agent will be trained.
        num_timesteps: int
            Total number of timesteps for which the agent will be trained.
        episode_length: int
            Maximum length of an episode.
        action_repeat: int, optional
            Number of times each action is repeated. Default is 1.
        num_envs: int, optional
            Number of parallel environments. Default is 1.
        num_eval_envs: int, optional
            Number of environments for evaluation. Default is 128.
        policy_lr: float, optional
            Learning rate for the policy network. Default is 1e-4.
        alpha_lr: float, optional
            Learning rate for the alpha parameter. Default is 1e-4.
        critic_lr: float, optional
            Learning rate for the critic network. Default is 1e-4.
        seed: int, optional
            Random seed for reproducibility. Default is 0.
        batch_size: int, optional
            Batch size for training. Default is 256.
        contrastive_loss_fn: str, optional
            Type of contrastive loss function. Default is "binary".
        energy_fn: str, optional
            Type of energy function. Default is "l2".
        logsumexp_penalty: float, optional
            Penalty for the log-sum-exp term in the loss function. Default is 0.0.
        l2_penalty: float, optional
            L2 regularization penalty. Default is 0.0.
        resubs: bool, optional
            Whether to use resubstitution in losses.py. Default is True.
        num_evals: int, optional
            Number of evaluation runs. Default is 1.
        min_replay_size: int, optional
            Minimum replay buffer size before starting training. Default is 0.
        max_replay_size: Optional[int], optional
            Maximum replay buffer size. Default is None.
        deterministic_eval: bool, optional
            If True, evaluation is deterministic. Default is False.
        progress_fn: Callable[[int, Metrics], None], optional
            Function to call to report progress. Default is a no-op lambda.
        checkpoint_logdir: Optional[str], optional
            Directory to save checkpoints. Default is None.
        eval_env: Optional[envs.Env], optional
            Evaluation environment. Default is None.
        unroll_length: int, optional
            Length of time to unroll the environment. Default is 50.
        train_step_multiplier: int, optional
            Number of SGD steps multiplier. Default is 1.
        config: NamedTuple, optional
            Configuration settings. Default is None.
        use_ln: bool, optional
            If True, use layer normalization. Default is False.
        h_dim: int, optional
            Dimension of the hidden layers. Default is 256.
        n_hidden: int, optional
            Number of hidden layers. Default is 2.
        repr_dim: int, optional
            Dimension of the representation from the state-action and goal encoders. Default is 64.
        visualization_interval: int, optional
            Number of evals between each visualization of trajectories. Default is 5.


    Raises:
        ValueError
            If the minimum replay size is greater than or equal to the number of timesteps.

    Returns:
        None

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
    print(f"Calculated env_steps_per_actor_step as action_repeat ({action_repeat}) * num_envs ({num_envs}) * unroll_length ({unroll_length}) = {env_steps_per_actor_step}")
    num_prefill_actor_steps = min_replay_size // unroll_length + 1
    print(f"Calculated num_prefill_actor_steps as min_replay_size ({min_replay_size}) // unroll_length ({unroll_length}) + 1 = {num_prefill_actor_steps}")
    print("Num_prefill_actor_steps: ", num_prefill_actor_steps)
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    print(f"Calculated num_prefill_env_steps as num_prefill_actor_steps ({num_prefill_actor_steps}) * env_steps_per_actor_step ({env_steps_per_actor_step}) = {num_prefill_env_steps}")
    print("Num prefill env steps:", num_prefill_env_steps)
    assert num_timesteps - min_replay_size >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    # Double negative with integer division acts as ceiling function: ceil(num_timesteps - num_prefill_env_steps / (num_evals_after_init * env_steps_per_actor_step))
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
    goal_size = obs_size - state_dim
    print("obs_size", obs_size)
    print("action_size", action_size)
    print("state_dim", state_dim)
    
    dummy_obs = jnp.zeros((obs_size,))
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
    
    saved_params = brax.io.model.load_params(saved_ckpt_path)
    expert_params, _, _ = saved_params
    expert_params = jax.lax.stop_gradient(expert_params)
    # Network functions
    block_size = 2 # Maybe make this a hyperparameter
    num_blocks = max(1, n_hidden // block_size)
    actor = Net(action_size * 2, h_dim, num_blocks, block_size, use_ln)
    expert = Net(action_size * 2, h_dim, num_blocks, block_size, use_ln)
    context_encoder = Net(goal_size * 2, h_dim, num_blocks, block_size, use_ln)
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size) # Would like to replace this but it's annoying to.

    # Initialize training state (not sure if it makes sense to split and fold local_key here)
    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)    
    training_state = _init_training_state(global_key, actor, context_encoder, var_post, env.state_dim, len(env.goal_indices), env.action_size, episode_length, policy_lr, alpha_lr, num_local_devices_to_use)
    del global_key
    
    # Update functions (may replace later: brax makes it opaque)
    alpha_update = gradients.gradient_update_fn(alpha_loss, training_state.alpha_state.tx, pmap_axis_name=_PMAP_AXIS_NAME)
    actor_update = gradients.gradient_update_fn(actor_loss, training_state.actor_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    if var_post == "meanfield":
        context_update = gradients.gradient_update_fn(context_loss_meanfield, training_state.context_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    else:
        context_update = gradients.gradient_update_fn(context_loss, training_state.context_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    
    def update_step(carry, transitions):
        training_state, key = carry
        key, key_alpha, key_actor = jax.random.split(key, 3)

        alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(training_state.alpha_state.params, actor, parametric_action_distribution, training_state, transitions, action_size,
                                                                       key_alpha, optimizer_state=training_state.alpha_state.opt_state)
        alpha = jnp.exp(alpha_params["log_alpha"])
        
        (actor_loss, actor_metrics), actor_params, actor_optimizer_state = actor_update(training_state.actor_state.params, training_state, actor, 
                                                                                        parametric_action_distribution, alpha, transitions, 
                                                                                        config, env.state_dim, env.goal_indices, energy_fn, 
                                                                                        key_actor, optimizer_state=training_state.actor_state.opt_state)

        metrics = {
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(alpha_params["log_alpha"]),
        }
        metrics.update(actor_metrics)

        new_training_state = TrainingState(
            env_steps=training_state.env_steps,
            gradient_steps=training_state.gradient_steps + 1,
            actor_state=training_state.actor_state.replace(params=actor_params, opt_state=actor_optimizer_state),
            alpha_state=training_state.alpha_state.replace(params=alpha_params, opt_state=alpha_optimizer_state),
            context_state=training_state.context_state,
        )
        return (new_training_state, key), metrics
    
        return (new_training_state, key), metrics
    
    def update_step_context(carry, trajectories):
        training_state, key = carry
        key, key_context = jax.random.split(key, 2)
        # print("trajectories shape in update context", trajectories.observation.shape)
        (context_loss_val, context_metrics), context_params, context_optimizer_state = context_update(
            training_state.context_state.params, 
            context_encoder, 
            trajectories, 
            env.state_dim, 
            env.goal_indices,
            repr_dim,  # Use same dimension as other representations
            0.01,  # Hyperparameter for KL regularization
            key_context, 
            optimizer_state=training_state.context_state.opt_state
        )

        metrics = {
            "context_loss": context_loss_val,
        }
        metrics.update(context_metrics)

        new_training_state = TrainingState(
            env_steps=training_state.env_steps,
            gradient_steps=training_state.gradient_steps,
            actor_state=training_state.actor_state,
            alpha_state=training_state.alpha_state,
            context_state=training_state.context_state.replace(params=context_params, opt_state=context_optimizer_state),
        )
        return (new_training_state, key), metrics
    

    def get_experience(env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            env_state, transition = actor_step(env, env_state, expert, parametric_action_distribution, expert_params, current_key, extra_fields=("truncation", "traj_id"))
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=unroll_length)
        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state
    
    def get_experience_context(actor_params, key):
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            env_state, transition = actor_step(env_train_context, env_state, actor, parametric_action_distribution, actor_params, current_key)
            return (env_state, next_key), transition

        # Looking at the original environment initialization to match the key structure
        # First, create appropriate keys for all environments
        reset_key, step_key = jax.random.split(key)
        
        # Create the exact same key structure as used in the main training loop
        # This is crucial - the key structure must match what the env expects
        env_keys = jax.random.split(reset_key, num_envs)
        
        # The environment expects a key with shape [num_envs, 2]
        # Each key is a PRNGKey with shape [2]
        print("env_keys shape:", env_keys.shape)
        
        # Reset the environment with properly structured keys
        env_state = env_train_context.reset(env_keys)
        
        print("get experience context env_state observation shape", env_state.obs.shape)
        (env_state, _), data = jax.lax.scan(f, (env_state, step_key), (), length=episode_length)
        # Flip axes 0 and 1 for all attributes in data
        # This transforms data from shape [unroll_length, num_envs, ...] to [num_envs, unroll_length, ...]
        print("data shape before flipping", data.observation.shape)
        data = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1) if isinstance(x, jnp.ndarray) and x.ndim >= 2 else x,
            data
        )
        print("data shape", data.observation.shape)
        return data

    def training_step(training_state, env_state, buffer_state, key):
        # Collect experience
        experience_key, training_key = jax.random.split(key, 2)
        env_state, buffer_state = get_experience(env_state, buffer_state, experience_key)
        training_state = training_state.replace(env_steps=training_state.env_steps + env_steps_per_actor_step)
        
        # Train
        training_state, buffer_state, metrics = train_steps(training_state, buffer_state, training_key)
        return training_state, env_state, buffer_state, metrics

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        def f(carry, unused):
            # Collect experience
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(env_state, buffer_state, key)
            new_training_state = training_state.replace(env_steps=training_state.env_steps + env_steps_per_actor_step)
            return (new_training_state, env_state, buffer_state, new_key), ()
        print("current num_prefill_actor_steps", num_prefill_actor_steps)
        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=num_prefill_actor_steps)[0]
    
    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

    def train_steps(training_state, buffer_state, key):
        # Sample, process, shuffle, then train
        ## Sample from buffer
        experience_key, training_key, sampling_key = jax.random.split(key, 3)
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        # print("TRANSITIONS SHAPE", transitions.observation.shape)
        
        ## Process
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        vmap_flatten_crl_fn = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, None, 0, 0))
        transitions = vmap_flatten_crl_fn(config, env, transitions, batch_keys)
        print("TRANSITIONS SHAPE AFTER PROCESSING", transitions.observation.shape)
        trajectories = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, num_envs, x.shape[1]) + x.shape[2:]), transitions)
        print("TRAJECTORIES SHAPE", trajectories.observation.shape)
        
        ## Shuffle transitions and reshape them into (number_of_sgd_steps, batch_size, ...)
        transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions)
        print("TRANSITIONS SHAPE AFTER RESHAPING", transitions.observation.shape)
        permutation = jax.random.permutation(experience_key, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]), transitions)
        # print("TRANSITIONS SHAPE AFTER SHUFFLING", transitions.observation.shape)
        
        ## Train
        (training_state, _), metrics1 = jax.lax.scan(update_step, (training_state, training_key), transitions)
        
        trajectories = get_experience_context(jax.lax.stop_gradient(training_state.actor_state.params), training_key)
        # Add an axis at the front of trajectories to match the expected shape for update_step_context
        trajectories = jax.tree_util.tree_map(lambda x: x[None, ...], trajectories)
        print("trajectories before slice", trajectories.observation.shape)
        trajectories = jax.tree_util.tree_map(
            lambda x: x[:, :, :-1, ...] if x.ndim > 2 else x,
            trajectories
        )
        print("trajectories before update step context", trajectories.observation.shape)
        
        (training_state, _), metrics2 = jax.lax.scan(update_step_context, (training_state, training_key), trajectories)
        metrics = {**metrics1, **metrics2}
        return training_state, buffer_state, metrics

    # def scan_train_steps(n, ts, bs, update_key):
    #     def body(carry, unsued_t):
    #         ts, bs, update_key = carry
    #         new_key, update_key = jax.random.split(update_key)
    #         ts, bs, metrics = train_steps(ts, bs, update_key)
    #         return (ts, bs, new_key), metrics
    #     return jax.lax.scan(body, (ts, bs, update_key), (), length=n)

    def training_epoch(training_state, env_state, buffer_state, key):
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key, update_key = jax.random.split(k, 3)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            # (ts, bs, update_key), _ = scan_train_steps(train_step_multiplier - 1, ts, bs, update_key)
            return (ts, es, bs, new_key), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=num_training_steps_per_epoch)
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(training_state, env_state, buffer_state, key):
        nonlocal training_walltime
        t = time.time()
        (training_state, env_state, buffer_state, metrics) = training_epoch(training_state, env_state, buffer_state, key)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
        # print("METRICS for logging********", metrics)
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return (training_state, env_state, buffer_state, metrics)

    # Initialization and setup
    ## Env init
    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)
    env_keys = jax.random.split(env_key, num_envs // jax.process_count())
    env_keys = jnp.reshape(env_keys, (num_local_devices_to_use, -1) + env_keys.shape[1:])
    # print("env keys", env_keys.shape)
    env_state = jax.pmap(env.reset)(env_keys)

    ## Replay buffer init and prefill
    buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, num_local_devices_to_use))
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, num_local_devices_to_use)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(training_state, env_state, buffer_state, prefill_keys)
    replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    # print("replay size after prefill", replay_size)
    # print("replay buffer.size", replay_buffer.size)
    # print("jax process count", jax.process_count())
    assert replay_size >= min_replay_size
    training_walltime = time.time() - t
    # Print the contents of the replay buffer after it has been initially filled
    buffer_contents = jax.tree_util.tree_map(lambda x: x.block_until_ready().shape, buffer_state)
    # print("Replay buffer contents after initial fill:", buffer_state.data.shape, buffer_state.insert_position, buffer_state.sample_position)
    # print(buffer_contents)

    ## Eval init
    global make_policy # Apparently necessary for train() to see make_policy
    make_policy = functools.partial(make_policy, actor, parametric_action_distribution)
    if not eval_env:
        eval_env = environment
    eval_env = TrajectoryIdWrapper(eval_env)
    eval_env = wrap_for_training(eval_env, episode_length=episode_length, action_repeat=action_repeat)
    evaluator = CrlEvaluator(eval_env, functools.partial(make_policy, deterministic=deterministic_eval), num_eval_envs=num_eval_envs,
                             episode_length=episode_length, action_repeat=action_repeat, key=eval_key)

    ## Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(_unpmap(training_state.actor_state.params), training_metrics={})
        logging.info(metrics)
        progress_fn(0, metrics, make_policy, _unpmap(training_state.actor_state.params), unwrapped_env)

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
            ## Save policy, critic, and context params
            if checkpoint_logdir:
                params = _unpmap((training_state.actor_state.params, training_state.context_state.params))
                path = f"{checkpoint_logdir}/step_{current_step}.pkl"
                brax.io.model.save_params(path, params)

            ## Run evals
            metrics = evaluator.run_evaluation(_unpmap(training_state.actor_state.params), training_metrics)
            logging.info(metrics)
            do_render = (eval_epoch_num % visualization_interval) == 0
            progress_fn(current_step, metrics, make_policy, _unpmap(training_state.actor_state.params), unwrapped_env, do_render)

    # Final validity checks
    ## Verify number of steps is sufficient
    total_steps = current_step
    logging.info("total steps: %s", total_steps)
    assert total_steps >= num_timesteps

    ## If there were no mistakes the training_state should still be identical on all devices
    pmap.assert_is_replicated(training_state)
    pmap.synchronize_hosts()
    
    params = _unpmap((training_state.actor_state.params, training_state.context_state.params))
    return (make_policy, params, metrics)

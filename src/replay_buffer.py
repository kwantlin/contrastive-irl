import functools
from typing import Generic, Tuple, TypeVar, Any, NamedTuple

import jax
from brax.training.acme.types import NestedArray
from brax.training.replay_buffers import ReplayBuffer, ReplayBufferState
from brax.training.types import PRNGKey
from jax import flatten_util, numpy as jnp

Sample = TypeVar("Sample")


class QueueBase(ReplayBuffer[ReplayBufferState, Sample], Generic[Sample]):
    """Base class for limited-size FIFO reply buffers.

    Implements an `insert()` method which behaves like a limited-size queue.
    I.e. it adds samples to the end of the queue and, if necessary, removes the
    oldest samples form the queue in order to keep the maximum size within the
    specified limit.

    Derived classes must implement the `sample()` method.
    """

    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample: Sample,
        sample_batch_size: int,
        num_envs: int,
        episode_length: int,
    ):
        self._flatten_fn = jax.vmap(jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0]))

        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(jax.vmap(self._unflatten_fn))
        data_size = len(dummy_flatten)

        self._data_shape = (max_replay_size, num_envs, data_size)
        self._data_dtype = dummy_flatten.dtype
        self._sample_batch_size = sample_batch_size
        self._size = 0
        self.num_envs = num_envs
        self.episode_length = episode_length

    def init(self, key: PRNGKey) -> ReplayBufferState:
        return ReplayBufferState(
            data=jnp.zeros(self._data_shape, self._data_dtype),
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def check_can_insert(self, buffer_state, samples, shards):
        """Checks whether insert operation can be performed."""
        assert isinstance(shards, int), "This method should not be JITed."
        insert_size = jax.tree_util.tree_flatten(samples)[0][0].shape[0] // shards
        if self._data_shape[0] < insert_size:
            raise ValueError(
                "Trying to insert a batch of samples larger than the maximum replay"
                f" size. num_samples: {insert_size}, max replay size"
                f" {self._data_shape[0]}"
            )
        self._size = min(self._data_shape[0], self._size + insert_size)

    def insert_internal(
        self, buffer_state: ReplayBufferState, samples: Sample
    ) -> ReplayBufferState:
        """Insert data in the replay buffer.

        Args:
          buffer_state: Buffer state
          samples: Sample to insert with a leading batch size.

        Returns:
          New buffer state.
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"buffer_state.data.shape ({buffer_state.data.shape}) "
                f"doesn't match the expected value ({self._data_shape})"
            )

        update = self._flatten_fn(samples)
        data = buffer_state.data

        # If needed, roll the buffer to make sure there's enough space to fit
        # `update` after the current position.
        position = buffer_state.insert_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll

        # Update the buffer and the control numbers.
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + len(update)) % (len(data) + 1)
        sample_position = jnp.maximum(0, buffer_state.sample_position + roll)

        return buffer_state.replace(
            data=data,
            insert_position=position,
            sample_position=sample_position,
        )
    def sample_internal(
        self, buffer_state: ReplayBufferState
    ) -> Tuple[ReplayBufferState, Sample]:
        raise NotImplementedError(f"{self.__class__}.sample() is not implemented.")

    def size(self, buffer_state: ReplayBufferState) -> int:
        return (
            buffer_state.insert_position - buffer_state.sample_position
        )  # pytype: disable=bad-return-type  # jax-ndarray


class Transition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray


class TrajectoryUniformSamplingQueue(QueueBase[Sample], Generic[Sample]):
    """Implements an uniform sampling limited-size replay queue BUT WITH TRAJECTORIES."""

    def sample_internal(self, buffer_state: ReplayBufferState) -> Tuple[ReplayBufferState, Sample]:
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})"
            )
        key, sample_key, shuffle_key = jax.random.split(buffer_state.key, 3)
        # NOTE: this is the number of envs to sample but it can be modified if there is OOM
        shape = self.num_envs

        # Sampling envs idxs
        envs_idxs = jax.random.choice(sample_key, jnp.arange(self.num_envs), shape=(shape,), replace=False)
        print("ENVS IDXS", envs_idxs.shape)

        @functools.partial(jax.jit, static_argnames=("rows", "cols"))
        def create_matrix(rows, cols, min_val, max_val, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            start_values = jax.random.randint(subkey, shape=(rows,), minval=min_val, maxval=max_val)
            row_indices = jnp.arange(cols)
            matrix = start_values[:, jnp.newaxis] + row_indices
            return matrix

        @jax.jit
        def create_batch(arr_2d, indices):
            return jnp.take(arr_2d, indices, axis=0, mode="wrap")

        create_batch_vmaped = jax.vmap(create_batch, in_axes=(1, 0))

        matrix = create_matrix(
            shape,
            self.episode_length,
            buffer_state.sample_position,
            buffer_state.insert_position - self.episode_length,
            sample_key,
        )

        batch = create_batch_vmaped(buffer_state.data[:, envs_idxs, :], matrix)
        transitions = self._unflatten_fn(batch)
        print("transitions", transitions.observation.shape)
        return buffer_state.replace(key=key), transitions

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["config", "env"])
    def flatten_crl_fn(config, env, transition: Transition, sample_key: PRNGKey) -> Transition:
        # print("FLATTENING CRL FN")
        goal_key, transition_key = jax.random.split(sample_key)

        # Because it's vmaped transition obs.shape is of shape (transitions,obs_dim)
        seq_len = transition.observation.shape[0]
        print("SEQ LEN", transition.observation.shape)
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
        discount = config.discounting ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
        probs = is_future_mask * discount
        single_trajectories = jnp.concatenate(
            [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len, axis=0
        )
        print("traj_id", transition.extras["state_extras"]["traj_id"].shape)
        print("SINGLE TRAJECTORIES", single_trajectories.shape)
        probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
        print("probs shape", probs.shape)
        goal_index = jax.random.categorical(goal_key, jnp.log(probs))
        print("goal index", goal_index)
        print("goal index shape", goal_index.shape)
        future_state = jnp.take(transition.observation, goal_index[:-1], axis=0)
        future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
        # goal portion of the future state
        goal = future_state[:, env.goal_indices]
        future_state = future_state[:, :env.state_dim]
        # current state
        state = transition.observation[:-1, :env.state_dim]
        state_goal_portion = state[:, env.goal_indices]
        new_obs = jnp.concatenate([state, goal], axis=1)
        target = transition.observation[:-1, env.state_dim:]
        truncation = transition.extras["state_extras"]["truncation"][:-1]
        mask = 1.0 - truncation.astype(jnp.float32)
        
        # Sample value_goals for HILP value loss: mix between trajectory goal and random goal
        value_goal_key, random_goal_key = jax.random.split(goal_key, 2)
        
        # Already have trajectory goal from goal_index (geometric sampling)
        traj_goal_obs = jnp.take(transition.observation, goal_index[:-1], axis=0)
        
        # Sample random goal: uniform over all timesteps in trajectory (0 to seq_len-1)
        random_goal_index = jax.random.randint(random_goal_key, shape=(seq_len - 1,), minval=0, maxval=seq_len)
        random_goal_obs = jnp.take(transition.observation, random_goal_index, axis=0)
        
        # Mix between trajectory goal and random goal based on probabilities
        # Default: value_p_trajgoal=0.625, value_p_randomgoal=0.375
        value_p_trajgoal = 0.625
        value_p_randomgoal = 0.375
        
        # Sample which type of goal to use for each timestep
        mix_key, _ = jax.random.split(value_goal_key)
        use_traj_goal = jax.random.bernoulli(mix_key, p=value_p_trajgoal, shape=(seq_len - 1,))
        value_goals = jnp.where(use_traj_goal[:, None], traj_goal_obs, random_goal_obs)
        
        # Compute relabeled_mask and relabeled_rewards
        # Check if current agent position matches the goal position from value_goals
        # Extract current agent positions (goal_indices refer to positions within the state vector)
        current_positions = state[:, env.goal_indices]
        # Extract goal positions from value_goals (value_goals are full observations)
        value_goal_state = value_goals[:, :env.state_dim]
        value_goal_positions = value_goal_state[:, env.goal_indices]
        # Check if current position is close to goal position (within threshold)
        goal_dist = jnp.linalg.norm(current_positions - value_goal_positions, axis=-1)
        success = (goal_dist < env.goal_reach_thresh).astype(jnp.float32)
        relabeled_mask = 1.0 - success
        relabeled_rewards = success - 1.0  # 0 if goal achieved, -1 otherwise (gc_negative=True)
        
        noise = transition.extras["state_extras"]["noise"][:-1] if "noise" in transition.extras["state_extras"].keys() else jnp.zeros(1)
        print("state, target, noise", state.shape, target.shape, noise.shape)
        next_state = transition.observation[1:, :env.state_dim]
        next_state_goal_portion = next_state[:, env.goal_indices]
        next_action = transition.action[1:]
        print("next state", next_state.shape)
        extras = {
            "policy_extras": {},
            "state_extras": {
                "truncation": jnp.squeeze(truncation),
                "mask": jnp.squeeze(mask),
                "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
                "noise": noise,
            },
            "mask": mask,
            "value_goals": value_goals,
            "relabeled_mask": relabeled_mask,
            "relabeled_rewards": relabeled_rewards,
            "state": state,
            "state_goal_portion": state_goal_portion,
            "future_state": future_state,
            "future_action": future_action,
            "target": target,
            "next_state": next_state,
            "next_state_goal_portion": next_state_goal_portion,
            "next_action": next_action,
        }

        return transition._replace(
            observation=jnp.squeeze(new_obs),
            action=jnp.squeeze(transition.action[:-1]),
            reward=jnp.squeeze(transition.reward[:-1]),
            discount=jnp.squeeze(transition.discount[:-1]),
            extras=extras,
        )

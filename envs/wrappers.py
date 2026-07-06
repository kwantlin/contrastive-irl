import jax
import jax.numpy as jnp
import brax
from brax import envs
import numpy as np
from typing import Tuple, Dict, Any, Optional
from brax.envs import Wrapper, PipelineEnv, State


class TrajectoryIdWrapper(Wrapper):
    def __init__(self, env: PipelineEnv):
        super().__init__(env)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["traj_id"] = jnp.zeros(rng.shape[:-1])
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info.keys():
            traj_id = state.info["traj_id"] + jnp.where(state.info["steps"], 0, 1)
        else:
            traj_id = state.info["traj_id"]
        state = self.env.step(state, action)
        state.info["traj_id"] = traj_id
        return state

class RandomTargetWrapper(brax.envs.Wrapper):
    """Wrapper that generates a random target during environment resets.
    
    This wrapper intercepts the reset method call and instead calls reset_with_target
    using a randomly generated target value. This ensures the environment starts with
    a different goal target each time, which can be useful for exploration and generalization.
    """

    def __init__(self, env, min_range=-10.0, max_range=10.0):
        """Initializes the wrapper with target generation parameters.
        
        Args:
            env: The environment to wrap.
            min_range: Minimum value for random target coordinates.
            max_range: Maximum value for random target coordinates.
                   
        Raises:
            AttributeError: If the wrapped environment doesn't support reset_with_target.
        """
        super().__init__(env)
        self._env = env
        self.min_range = min_range
        self.max_range = max_range
        
        # Check if the environment supports reset_with_target
        if not hasattr(self._env, 'reset_with_target'):
            raise AttributeError(
                f"Environment {type(self._env).__name__} does not support reset_with_target. "
                "RandomTargetWrapper can only be used with environments that support this method."
            )
        
        # Get goal dimension from environment if available
        self.goal_dim = getattr(env, 'goal_dim', 2)
        print(f"RandomTargetWrapper initialized with goal dimension: {self.goal_dim}")
    
    def _generate_random_target(self, rng):
        """Generates a random target using the provided RNG key.
        
        Args:
            rng: JAX random key for generating the target.
            
        Returns:
            A tuple of (new_rng, random_target).
        """
        rng, target_key = jax.random.split(rng)
        random_target = jax.random.uniform(
            target_key, 
            shape=(self.goal_dim,), 
            minval=self.min_range, 
            maxval=self.max_range
        )
        return rng, random_target
    
    def reset(self, rng):
        """Reset the environment with a random target.
        
        Args:
            rng: JAX random key or batch of keys for resetting the environment.
            
        Returns:
            The state from reset_with_target.
        """
        try:
            # Handle both single and batched resets
            if hasattr(rng, 'shape') and len(rng.shape) > 1:
                # For batched resets, we need to generate a batch of targets
                batch_size = rng.shape[0]
                batch_rngs = jax.random.split(rng, batch_size + 1)
                parent_rng = batch_rngs[0]
                target_rngs = batch_rngs[1:]
                
                # Generate a random target for each environment in the batch
                random_targets = jax.vmap(lambda key: jax.random.uniform(
                    key, 
                    shape=(self.goal_dim,), 
                    minval=self.min_range, 
                    maxval=self.max_range
                ))(target_rngs)
                
                return self._env.reset_with_target(parent_rng, random_targets)
            else:
                # For single resets
                rng, random_target = self._generate_random_target(rng)
                return self._env.reset_with_target(rng, random_target)
        except Exception as e:
            print(f"Error in RandomTargetWrapper.reset: {e}")
            # Fall back to standard reset if something goes wrong
            return self._env.reset(rng)
    
    def reset_with_target(self, rng, target):
        """Allow explicit specification of target if needed.
        
        Args:
            rng: JAX random key or batch of keys for resetting the environment.
            target: The target to use for this reset.
            
        Returns:
            The state from reset_with_target.
        """
        return self._env.reset_with_target(rng, target)

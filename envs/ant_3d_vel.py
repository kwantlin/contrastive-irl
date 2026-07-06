import os
from typing import Tuple

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jnp
import mujoco

# This is based on original Ant environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/ant.py


class Ant3D(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 2.5),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        dense_reward: bool = False,
        randomize_start=False,
        goal_distance=10,
        target_vel_range: Tuple[float, float] = (0.0, 5.0),
        **kwargs,
    ):
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "assets", "ant_3d.xml"
        )
        sys = mjcf.load(path)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        if backend == "positional":
            # TODO: does the same actuator strength work as in spring
            sys = sys.replace(actuator=sys.actuator.replace(gear=200 * jnp.ones_like(sys.actuator.gear)))

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self.dense_reward = dense_reward
        self.state_dim = 29
        self.goal_indices = jnp.array([0, 1, 2])
        self.goal_reach_thresh = 0.5
        self.goal_distance = goal_distance
        self.randomize_start = randomize_start
        self._target_vel_range = target_vel_range

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # set the target q, qd
        rng, target_pos = self._random_target(rng)
        rng, target_vel = self._random_target_velocity(rng)
        q = q.at[-3:].set(target_pos)
        qd = qd.at[-3:].set(target_vel)

        if self.randomize_start:
            _, start_delta = self._random_target(rng)
            start = target_pos + start_delta
            q = q.at[:3].set(start)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        return state
    
    def reset_with_target(self, rng: jax.Array, target) -> State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # set the target q, qd
        # _, target = self._random_target(rng)
        q = q.at[-3:].set(target)
        qd = qd.at[-3:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # Enforce that target's z-position is constant and z-velocity is zero,
        # while allowing x and y to be determined by physics.
        q = pipeline_state.q.at[-1].set(pipeline_state0.q[-1])
        qd = pipeline_state.qd.at[-1].set(0.0)
        x_pos = pipeline_state.x.pos.at[-1, 2].set(pipeline_state0.x.pos[-1, 2])

        pipeline_state = pipeline_state.replace(
            q=q, qd=qd, x=pipeline_state.x.replace(pos=x_pos)
        )

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = 0.0

        old_obs = self._get_obs(pipeline_state0)
        old_dist = jnp.linalg.norm(old_obs[:3] - old_obs[-3:])
        obs = self._get_obs(pipeline_state)
        dist = jnp.linalg.norm(obs[:3] - obs[-3:])
        vel_to_target = (old_dist - dist) / self.dt
        success = jnp.array(dist < self.goal_reach_thresh, dtype=float)
        success_easy = jnp.array(dist < 2.0, dtype=float)

        if self.dense_reward:
            reward = 10 * vel_to_target + healthy_reward - ctrl_cost - contact_cost
        else:
            reward = success
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy,
        )
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        # remove target q, qd
        qpos = pipeline_state.q[:-3]
        qvel = pipeline_state.qd[:-3]

        target_pos = pipeline_state.x.pos[-1][:3]

        if self._exclude_current_positions_from_observation:
            qpos = qpos[3:]

        return jnp.concatenate([qpos] + [qvel] + [target_pos])

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a target location in a random circle slightly above xy plane."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        ang = jnp.pi * 2.0 * jax.random.uniform(rng2)
        target_x = self.goal_distance * jnp.cos(ang)
        target_y = self.goal_distance * jnp.sin(ang)
        target_z = jax.random.uniform(
            rng1, minval=self._healthy_z_range[0], maxval=self._healthy_z_range[1]
        )
        return rng, jnp.array([target_x, target_y, target_z])

    def _random_target_velocity(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a random target velocity in the xy plane."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, high = self._target_vel_range
        speed = jax.random.uniform(rng1, minval=low, maxval=high)
        ang = jnp.pi * 2.0 * jax.random.uniform(rng2)
        vx = speed * jnp.cos(ang)
        vy = speed * jnp.sin(ang)
        return rng, jnp.array([vx, vy, 0.0])

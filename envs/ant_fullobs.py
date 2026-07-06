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


class AntFullObs(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.0, 4.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        dense_reward: bool = False,
        randomize_start=False,
        goal_distance=10,
        # New reward weights for full observation task
        pos_reward_weight=3.0,
        # rot_reward_weight=0.5,
        vel_reward_weight=0.1,
        ang_reward_weight=0.05,
        # New success thresholds
        pos_reach_thresh=0.5,
        # rot_reach_thresh=0.2,
        vel_reach_thresh=0.15,
        ang_vel_reach_thresh=0.15,
        target_vel_range=(-1.0, 1.0),
        target_ang_vel_range=(-1.5, 1.5),
        **kwargs,
    ):
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "assets", "ant_fullobs.xml"
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
        self.goal_indices = jnp.array([0, 1, 2, 15, 16, 17, 18, 19, 20])
        self.goal_reach_thresh = 5
        self.goal_distance = goal_distance
        self.randomize_start = randomize_start
        self.pos_reward_weight = pos_reward_weight
        # self.rot_reward_weight = rot_reward_weight
        self.vel_reward_weight = vel_reward_weight
        self.ang_reward_weight = ang_reward_weight
        self.state_dim = 29  # 3 pos, 4 rot, 3 lin_vel, 3 ang_vel, 8 joints, 8 joint_vel
        self.pos_reach_thresh = pos_reach_thresh
        # self.rot_reach_thresh = rot_reach_thresh
        self.vel_reach_thresh = vel_reach_thresh
        self.ang_vel_reach_thresh = ang_vel_reach_thresh
        self.target_vel_range = target_vel_range
        self.target_ang_vel_range = target_ang_vel_range

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng_target = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(q, qd)

        # Generate and store a random target.
        rng, target_pos, target_vel, target_ang_vel = self._random_target(
            rng_target
        )
        target = {
            "pos": target_pos,
            # "rot": target_rot,
            "vel": target_vel,
            "ang_vel": target_ang_vel,
        }

        obs = self._get_obs(pipeline_state, target)
        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_pos": zero,
            # "reward_rot": zero,
            "reward_vel": zero,
            "reward_ang": zero,
            "reward_ctrl": zero,
            "reward_survive": zero,
            "vel_to_target": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
            "success_pos": zero,
            # "success_rot": zero,
            "success_vel": zero,
            "success_ang": zero,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(target=target)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        target = state.info["target"]

        pipeline_state = self.pipeline_step(pipeline_state0, action)

        def calculate_dist(p_state: base.State):
            torso_pos = p_state.q[:3]
            # torso_rot = p_state.q[3:7]
            torso_vel = p_state.qd[:3]
            torso_ang_vel = p_state.qd[3:6]

            pos_error = jnp.linalg.norm(torso_pos - target["pos"])
            # rot_error_quat = math.quat_mul(target["rot"], math.quat_inv(torso_rot))
            # rot_error_angle = 2 * jnp.arccos(jnp.clip(rot_error_quat[0], -1.0, 1.0))
            vel_error = jnp.linalg.norm(torso_vel - target["vel"])
            ang_vel_error = jnp.linalg.norm(torso_ang_vel - target["ang_vel"])

            dist = (
                self.pos_reward_weight * pos_error
                # + self.rot_reward_weight * rot_error_angle
                + self.vel_reward_weight * vel_error
                + self.ang_reward_weight * ang_vel_error
            )
            return dist, pos_error, vel_error, ang_vel_error

        dist, pos_err,  vel_err, ang_err = calculate_dist(pipeline_state)
        old_dist, _, _, _ = calculate_dist(pipeline_state0)

        vel_to_target = (old_dist - dist) / self.dt
        success = jnp.array(dist < self.goal_reach_thresh, dtype=float)
        success_easy = jnp.array(dist < 2.0 * self.goal_reach_thresh, dtype=float)

        success_pos = jnp.array(pos_err < self.pos_reach_thresh, dtype=float)
        # success_rot = jnp.array(rot_err < self.rot_reach_thresh, dtype=float)
        success_vel = jnp.array(vel_err < self.vel_reach_thresh, dtype=float)
        success_ang = jnp.array(ang_err < self.ang_vel_reach_thresh, dtype=float)

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = 0.0

        if self.dense_reward:
            reward = vel_to_target + healthy_reward - ctrl_cost - contact_cost
        else:
            reward = success

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        obs = self._get_obs(pipeline_state, target)

        state.metrics.update(
            reward_pos=-pos_err,
            # reward_rot=-rot_err,
            reward_vel=-vel_err,
            reward_ang=-ang_err,
            reward_ctrl=-ctrl_cost,
            reward_survive=healthy_reward,
            vel_to_target=vel_to_target,
            dist=dist,
            success=success,
            success_easy=success_easy,
            success_pos=success_pos,
            # success_rot=success_rot,
            success_vel=success_vel,
            success_ang=success_ang,
        )
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State, target: dict) -> jax.Array:
        """Observe ant's state and the full target state."""
        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            qpos = qpos[3:]  # remove x,y,z from observation

        target_obs = jnp.concatenate(
            [target["pos"], target["vel"], target["ang_vel"]]
        )

        return jnp.concatenate([qpos, qvel, target_obs])

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, ...]:
        """Returns a random target pose and velocities."""
        rng, rng_pos, rng_rot, rng_vel, rng_ang_vel = jax.random.split(rng, 5)

        # Sample a random radius and angle for the xy-plane
        rng_xy, rng_z = jax.random.split(rng_pos)
        rng_rad, rng_ang = jax.random.split(rng_xy)

        radius = jax.random.uniform(rng_rad, minval=0, maxval=self.goal_distance)
        ang = jnp.pi * 2.0 * jax.random.uniform(rng_ang)
        target_x = radius * jnp.cos(ang)
        target_y = radius * jnp.sin(ang)

        # Random z position within the healthy range
        target_z = jax.random.uniform(
            rng_z, minval=self._healthy_z_range[0]+0.2, maxval=0.5*self._healthy_z_range[1]
        )
        target_pos = jnp.array([target_x, target_y, target_z])

        # Random rotation
        # target_rot = jax.random.normal(rng_rot, (4,))
        # target_rot /= jnp.linalg.norm(target_rot)

        # Random linear velocity
        min_vel, max_vel = self.target_vel_range
        target_vel = jax.random.uniform(rng_vel, (3,), minval=min_vel, maxval=max_vel)

        # Random angular velocity
        min_ang_vel, max_ang_vel = self.target_ang_vel_range
        target_ang_vel = jax.random.uniform(
            rng_ang_vel, (3,), minval=min_ang_vel, maxval=max_ang_vel
        )

        return rng, target_pos, target_vel, target_ang_vel

import gym

from .trail import TrailEnv, TrailContinuousEnv

if "TrailTracker-v0" not in gym.envs.registry.env_specs:
    gym.envs.register(
        id="TrailTracker-v0",
        entry_point="zoo.envs:TrailEnv",
        max_episode_steps=200,
        reward_threshold=2000
    )

    gym.envs.register(
        id="TrailTrackerCont-v0",
        entry_point="zoo.envs:TrailContinuousEnv",
        max_episode_steps=200,
        reward_threshold=2000
    )

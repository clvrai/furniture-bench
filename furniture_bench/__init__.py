from gym.envs.registration import register

import isaacgym


register(
    id="Furniture-Image-Dummy-v0",
    entry_point="furniture_bench.envs.policy_envs.furniture_image_dummy_env:FurnitureImageDummyEnv",
    max_episode_steps=5000,
)

register(
    id="Furniture-Image-v0",
    entry_point="furniture_bench.envs.policy_envs.furniture_image_env:FurnitureImageEnv",
    max_episode_steps=5000,
)

# Environment that gives pre-trainedimage features as observation.
register(
    id="Furniture-Image-Feature-Dummy-v0",
    entry_point="furniture_bench.envs.policy_envs.furniture_image_feature_dummy_env:FurnitureImageFeatureDummyEnv",
    max_episode_steps=5000,
)

register(
    id="Furniture-Image-Feature-v0",
    entry_point="furniture_bench.envs.policy_envs.furniture_image_feature_env:FurnitureImageFeatureEnv",
    max_episode_steps=5000,
)

register(
    id="Furniture-Image-Feature-Sim-v0",
    entry_point="furniture_bench.envs.policy_envs.furniture_image_feature_sim_env:FurnitureImageFeatureSimEnv",
    max_episode_steps=600,
)

register(
    id="Furniture-Sim-Env-v0",
    entry_point="furniture_bench.envs.furniture_sim_env:FurnitureSimEnv",
    max_episode_steps=5000,
)

register(
    id="Furniture-Env-v0",
    entry_point="furniture_bench.envs.furniture_env:FurnitureEnv",
    max_episode_steps=5000,
)

register(
    id="Furniture-Image-Sim-Env-v0",
    entry_point="furniture_bench.envs.policy_envs.furniture_image_sim_env:FurnitureImageSimEnv",
    max_episode_steps=600,
)

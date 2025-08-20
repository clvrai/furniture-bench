"""
Register FurnitureBench and FurnitureSim environments to OpenAI Gym.
"""

from gymnasium.envs.registration import register

# Ignore ImportError from isaacgym.
try:
    import isaacgym
except ImportError:
    pass


# FurnitureBench environment with full observation.
register(
    id="FurnitureBench-v0",
    entry_point="furniture_bench.envs.furniture_bench_env:FurnitureBenchEnv",
)

# FurnitureBench with 224x224 image observation.
register(
    id="FurnitureBenchImage-v0",
    entry_point="furniture_bench.envs.policy_envs.furniture_bench_image:FurnitureBenchImage",
)

# FurnitureBench with R3M or VIP image feature observation.
register(
    id="FurnitureBenchImageFeature-v0",
    entry_point="furniture_bench.envs.policy_envs.furniture_bench_image_feature:FurnitureBenchImageFeature",
)

# FurnitureSim environment.
register(
    id="FurnitureSim-v0",
    entry_point="furniture_bench.envs.furniture_sim_env:FurnitureSimEnv",
)

register(
    id="FurnitureSimLegacy-v0",
    entry_point="furniture_bench.envs.legacy_envs.furniture_sim_legacy_env:FurnitureSimEnvLegacy",
)

# FurnitureSim environment with all available observation.
register(
    id="FurnitureSimFull-v0",
    entry_point="furniture_bench.envs.furniture_sim_env:FurnitureSimFullEnv",
)

# FurnitureSim with R3M or VIP image feature observation.
register(
    id="FurnitureSimImageFeature-v0",
    entry_point="furniture_bench.envs.policy_envs.furniture_sim_image_feature:FurnitureSimImageFeature",
)

# FurnitureSim environment with state observations.
register(
    id="FurnitureSimState-v0",
    entry_point="furniture_bench.envs.furniture_sim_env:FurnitureSimStateEnv",
)

# Dummy environments only define observation and action spaces.
register(
    id="FurnitureDummy-v0",
    entry_point="furniture_bench.envs.policy_envs.furniture_dummy:FurnitureDummy",
)

register(
    id="FurnitureImageFeatureDummy-v0",
    entry_point="furniture_bench.envs.policy_envs.furniture_image_feature_dummy:FurnitureImageFeatureDummy",
)

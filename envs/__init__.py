import logging
from gymnasium.envs.registration import register

register(
    id="RescueWorld-v0",
    entry_point="envs.rescueworld.rw:RescueWorld",
)

try:
    import adept_envs

    register(
        id="kitchen-all-v0",
        entry_point="envs.kitchen.v0:KitchenAllV0",
        max_episode_steps=500,
        reward_threshold=1.0,
    )

except ImportError:
    logging.warning("Kitchen not installed, skipping")


try:
    register(
        id="libero-goal-v0",
        entry_point="envs.libero.libero_goal:ConcatControlEnv",
        max_episode_steps=5,
        reward_threshold=1.0,
    )
except ImportError:
    logging.warning("Libero not installed, skipping")
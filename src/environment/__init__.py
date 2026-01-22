import gymnasium
from gymnasium.envs.registration import register

from .solar_merchant_env import SolarMerchantEnv, load_environment

# Register environment with Gymnasium
register(
    id='SolarMerchant-v0',
    entry_point='src.environment.solar_merchant_env:SolarMerchantEnv',
)

__all__ = ['SolarMerchantEnv', 'load_environment']

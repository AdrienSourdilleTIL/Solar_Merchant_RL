import gymnasium
from gymnasium.envs.registration import register

from .solar_merchant_env import SolarMerchantEnv, load_environment
from .solar_plant import PlantConfig, Battery, Settlement, DataManager
from .battery_env import BatteryEnv, load_battery_env
from .commitment_env import CommitmentEnv, load_commitment_env
from .hierarchical_orchestrator import HierarchicalOrchestrator, load_orchestrator

# Register environments with Gymnasium
register(
    id='SolarMerchant-v0',
    entry_point='src.environment.solar_merchant_env:SolarMerchantEnv',
)

register(
    id='SolarMerchantBattery-v0',
    entry_point='src.environment.battery_env:BatteryEnv',
)

register(
    id='SolarMerchantCommitment-v0',
    entry_point='src.environment.commitment_env:CommitmentEnv',
)

__all__ = [
    # Original monolithic environment
    'SolarMerchantEnv',
    'load_environment',
    # Shared components
    'PlantConfig',
    'Battery',
    'Settlement',
    'DataManager',
    # Hierarchical environments
    'BatteryEnv',
    'load_battery_env',
    'CommitmentEnv',
    'load_commitment_env',
    # Orchestrator
    'HierarchicalOrchestrator',
    'load_orchestrator',
]

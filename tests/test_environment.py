import pytest
import numpy as np
from gymnasium import spaces

from auction_gym.envs.rtb_env import RTBAuctionEnv
from auction_gym.core.agent import LinearAgent
from auction_gym.core.valuation import DeterministicValuation
from auction_gym.core.auction import SecondPriceAuction
from auction_gym.core.utility import LinearUtility, BaseUtility


class RiskAverseUtility(BaseUtility):
    """A custom utility model for testing."""
    def __call__(self, won: bool, valuation: float, payment: float) -> float:
        if won:
            return np.log(1 + valuation - payment)
        return 0.0

@pytest.fixture
def basic_env():
    """A basic environment with two simple agents for testing."""
    agents = [
        LinearAgent(
            action_space=spaces.Box(low=0, high=1, shape=(1,)),
            valuation=DeterministicValuation(0.8),
            lambda_param=1.0  # Bids 0.8
        ),
        LinearAgent(
            action_space=spaces.Box(low=0, high=1, shape=(1,)),
            valuation=DeterministicValuation(0.6),
            lambda_param=1.0  # Bids 0.6
        )
    ]
    env = RTBAuctionEnv(agents=agents, auction_mechanism=SecondPriceAuction())
    return env

def test_env_initialization(basic_env):
    """Test that the environment initializes correctly."""
    assert basic_env.n_agents == 2
    action_space = basic_env.action_spaces['agent_0']
    assert isinstance(action_space, spaces.Box)
    assert action_space.shape == (1,)

def test_env_reset(basic_env):
    """Test the reset method."""
    obs, info = basic_env.reset()
    assert 'agent_0' in obs and 'agent_1' in obs
    agent_obs = obs['agent_0']
    assert isinstance(agent_obs, np.ndarray)
    assert isinstance(info, dict)
    assert basic_env.current_step == 0

def test_env_step_second_price(basic_env):
    """Test a single step with a second-price auction."""
    env = basic_env
    obs, info = env.reset()
    
    # Agents will use their act() methods - LinearAgent bids lambda * valuation
    # Agent 0: lambda=1.0 * valuation=0.8 = 0.8
    # Agent 1: lambda=1.0 * valuation=0.6 = 0.6
    # Since these are non-trainable agents, we don't provide action_dict
    action_dict = {}  # Empty action dict for non-trainable agents
    obs, rewards, terminated, truncated, info = env.step(action_dict)
    
    # Check that info contains the expected auction information
    agent_info = info['agent_0']  # All agents get the same info
    
    # Winner should be agent 0 (higher bid of 0.8)
    assert agent_info['winner'] == 0
    # Payment should be the second-highest bid (0.6)
    assert agent_info['payment'] == 0.6
    
    # Rewards (using LinearUtility: valuation - payment if won, 0 if lost)
    # Agent 0 (winner): 0.8 - 0.6 = 0.2
    # Agent 1 (loser): 0.0
    expected_rewards = {'agent_0': 0.2, 'agent_1': 0.0}
    assert np.allclose(rewards['agent_0'], expected_rewards['agent_0'])
    assert np.allclose(rewards['agent_1'], expected_rewards['agent_1'])
    
    assert env.current_step == 1

def test_env_with_custom_utility():
    """Test the environment with a custom utility model."""
    agents = [
        LinearAgent(
            action_space=spaces.Box(low=0, high=1, shape=(1,)), 
            valuation=DeterministicValuation(0.9),
            utility=RiskAverseUtility()  # Custom utility for first agent
        ),
        LinearAgent(
            action_space=spaces.Box(low=0, high=1, shape=(1,)), 
            valuation=DeterministicValuation(0.5),
            utility=LinearUtility()  # Default utility for second agent
        )
    ]
    env = RTBAuctionEnv(agents=agents)
    obs, _ = env.reset()
    
    # Step with empty action dict (non-trainable agents use their act() method)
    _, rewards, _, _, info = env.step({})
    
    # Agent 0 wins with bid 0.9 and pays 0.5 (second highest bid)
    # Utility for agent 0 is log(1 + 0.9 - 0.5) = log(1.4)
    expected_reward_0 = np.log(1.4)
    # Utility for agent 1 is 0 (didn't win)
    
    assert np.allclose(rewards['agent_0'], expected_reward_0)
    assert np.allclose(rewards['agent_1'], 0.0)
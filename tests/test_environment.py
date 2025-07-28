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
    assert isinstance(basic_env.action_space, spaces.Box)
    assert basic_env.action_space.shape == (2,)

def test_env_reset(basic_env):
    """Test the reset method."""
    obs, info = basic_env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    assert basic_env.current_step == 0

def test_env_step_second_price(basic_env):
    """Test a single step with a second-price auction."""
    env = basic_env
    obs, info = env.reset()
    
    # Agents bid their deterministic valuations (0.8 and 0.6)
    actions = np.array([agent.act(obs) for agent in env.agents])
    
    next_obs, rewards, terminated, truncated, info = env.step(actions)

    # Winner should be agent 0
    assert info['winner'] == 0
    # Payment should be the second-highest bid (0.6)
    assert info['payment'] == 0.6
    
    # Rewards (using LinearUtility: valuation - payment)
    # Agent 0 (winner): 0.8 - 0.6 = 0.2
    # Agent 1 (loser): 0.0
    assert np.allclose(rewards, [0.2, 0.0])
    
    assert env.current_step == 1

def test_env_with_custom_utility():
    """Test the environment with a custom utility model."""
    agents = [
        LinearAgent(
            spaces.Box(0,1,(1,)), 
            DeterministicValuation(0.9),
            utility=RiskAverseUtility()  # Custom utility for first agent
        ),
        LinearAgent(
            spaces.Box(0,1,(1,)), 
            DeterministicValuation(0.5),
            utility=LinearUtility()  # Default utility for second agent
        )
    ]
    env = RTBAuctionEnv(agents=agents)
    obs, _ = env.reset()
    
    actions = np.array([agent.act(obs) for agent in env.agents])
    _, rewards, _, _, info = env.step(actions)
    
    # Agent 0 wins and pays 0.5
    # Utility for agent 0 is log(1 + 0.9 - 0.5) = log(1.4)
    expected_reward_0 = np.log(1.4)
    # Utility for agent 1 is 0
    
    assert np.allclose(rewards, [expected_reward_0, 0.0])
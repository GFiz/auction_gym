import pytest
import numpy as np
from gymnasium import spaces
from auction_gym.core.agent import RandomAgent, LinearAgent
from auction_gym.core.valuation import DeterministicValuation

@pytest.fixture
def action_space():
    """A standard action space for testing."""
    return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

@pytest.fixture
def obs():
    """A standard observation for testing."""
    return np.array([0.5], dtype=np.float32)

def test_random_agent(action_space, obs):
    """Test that the RandomAgent returns a bid within the action space."""
    valuation = DeterministicValuation(0.8)
    agent = RandomAgent(action_space, valuation)
    
    # Random agent should ignore valuation and sample from the action space
    bid = agent.act(obs)
    assert action_space.low[0] <= bid <= action_space.high[0]

def test_linear_agent_truthful_bidding(action_space, obs):
    """Test LinearAgent with lambda=1.0 (truthful bidding)."""
    valuation = DeterministicValuation(0.75)
    agent = LinearAgent(action_space, valuation, lambda_param=1.0)
    
    # Bid should be exactly the valuation
    bid = agent.act(obs)
    assert bid == pytest.approx(0.75)

def test_linear_agent_shaded_bidding(action_space, obs):
    """Test LinearAgent with lambda<1.0 (shaded bidding)."""
    valuation = DeterministicValuation(0.8)
    agent = LinearAgent(action_space, valuation, lambda_param=0.5)
    
    # Bid should be 0.5 * 0.8 = 0.4
    bid = agent.act(obs)
    assert bid == pytest.approx(0.4)

def test_linear_agent_aggressive_bidding(action_space, obs):
    """Test LinearAgent with lambda>1.0 (aggressive bidding)."""
    valuation = DeterministicValuation(0.6)
    agent = LinearAgent(action_space, valuation, lambda_param=1.5)
    
    # Bid should be 1.5 * 0.6 = 0.9
    bid = agent.act(obs)
    assert bid == pytest.approx(0.9)

def test_linear_agent_clipping(obs):
    """Test that LinearAgent's bids are clipped to the action space."""
    # Action space is [0, 0.5]
    action_space = spaces.Box(low=0.0, high=0.5, shape=(1,), dtype=np.float32)
    valuation = DeterministicValuation(0.8)
    agent = LinearAgent(action_space, valuation, lambda_param=1.0) # Truthful bid would be 0.8
    
    # The bid should be clipped to the action space's high of 0.5
    bid = agent.act(obs)
    assert bid == pytest.approx(0.5) 
import pytest
import numpy as np
from gymnasium import spaces
from unittest.mock import Mock, patch

from auction_gym.envs.auction_env import AuctionEnv
from auction_gym.core.mechanism import SecondPriceAuction, FirstPriceAuction
from auction_gym.core.valuation import DeterministicValuation, UniformValuation
from auction_gym.core.utility import LinearUtility, RiskAverseUtility


class TestAuctionEnvInitialization:
    """Test AuctionEnv initialization and validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # Create test dictionaries with consistent keys
        self.agent_keys = ["agent_0", "agent_1"]
        self.valuations = {
            "agent_0": DeterministicValuation(0.8),
            "agent_1": DeterministicValuation(0.6)
        }
        self.utilities = {
            "agent_0": LinearUtility(),
            "agent_1": LinearUtility()
        }
        self.action_spaces = {
            "agent_0": self.action_space,
            "agent_1": self.action_space
        }
        self.observation_spaces = {
            "agent_0": self.obs_space,
            "agent_1": self.obs_space
        }
    
    def test_valid_initialization(self):
        """Test that AuctionEnv initializes correctly with valid parameters."""
        env = AuctionEnv(
            valuations=self.valuations,
            utilities=self.utilities,
            action_spaces=self.action_spaces,
            observation_spaces=self.observation_spaces
        )
        
        assert env.n_agents == 2
        assert env.agents == ["agent_0", "agent_1"]
        assert env.current_step == 0
        assert env.max_steps == 100
        assert isinstance(env.mechanism, SecondPriceAuction)
    
    def test_mismatched_keys_raises_error(self):
        """Test that initialization fails when dictionaries have different keys."""
        # Create mismatched dictionaries
        mismatched_valuations = {
            "agent_0": DeterministicValuation(0.8),
            "agent_2": DeterministicValuation(0.6)  # Different key
        }
        
        with pytest.raises(ValueError, match="All dictionaries must have the same agent keys"):
            AuctionEnv(
                valuations=mismatched_valuations,
                utilities=self.utilities,
                action_spaces=self.action_spaces,
                observation_spaces=self.observation_spaces
            )
    
    def test_custom_mechanism(self):
        """Test that custom mechanism can be specified."""
        env = AuctionEnv(
            valuations=self.valuations,
            utilities=self.utilities,
            action_spaces=self.action_spaces,
            observation_spaces=self.observation_spaces,
            mechanism=FirstPriceAuction()
        )
        
        assert isinstance(env.mechanism, FirstPriceAuction)
    
    def test_custom_max_steps(self):
        """Test that custom max_steps can be specified."""
        env = AuctionEnv(
            valuations=self.valuations,
            utilities=self.utilities,
            action_spaces=self.action_spaces,
            observation_spaces=self.observation_spaces,
            max_steps=50
        )
        
        assert env.max_steps == 50


class TestAuctionEnvReset:
    """Test AuctionEnv reset functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.agent_keys = ["agent_0", "agent_1"]
        self.valuations = {
            "agent_0": DeterministicValuation(0.8),
            "agent_1": DeterministicValuation(0.6)
        }
        self.utilities = {
            "agent_0": LinearUtility(),
            "agent_1": LinearUtility()
        }
        self.action_spaces = {
            "agent_0": self.action_space,
            "agent_1": self.action_space
        }
        self.observation_spaces = {
            "agent_0": self.obs_space,
            "agent_1": self.obs_space
        }
        
        self.env = AuctionEnv(
            valuations=self.valuations,
            utilities=self.utilities,
            action_spaces=self.action_spaces,
            observation_spaces=self.observation_spaces
        )
    
    def test_reset_initial_state(self):
        """Test that reset returns to initial state."""
        # Run a few steps first
        self.env.current_step = 5
        self.env.episode_data["bids"] = [[1.0, 2.0]]
        
        obs_dict, info_dict = self.env.reset()
        
        # Check state is reset
        assert self.env.current_step == 0
        assert self.env.episode_data["bids"] == []
        assert self.env.episode_data["valuations"] == []
        assert self.env.episode_data["allocations"] == []
        assert self.env.episode_data["payments"] == []
        
        # Check return format
        assert set(obs_dict.keys()) == {"agent_0", "agent_1"}
        
        # Check that info_dict contains the correct top-level keys (from AuctionEnv._get_info)
        assert set(info_dict.keys()) == {"step", "max_steps", "n_agents", "mechanism_type"}
        
        # Check observation format - should be numpy arrays, not dictionaries
        for obs in obs_dict.values():
            assert isinstance(obs, np.float64)
            assert obs.shape == ()
        
        
    
    def test_reset_with_seed(self):
        """Test that reset works with seed parameter."""
        obs_dict, info_dict = self.env.reset(seed=42)
        
        assert self.env.current_step == 0
        assert set(obs_dict.keys()) == {"agent_0", "agent_1"}


class TestAuctionEnvStep:
    """Test AuctionEnv step functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.agent_keys = ["agent_0", "agent_1"]
        self.valuations = {
            "agent_0": DeterministicValuation(0.8),
            "agent_1": DeterministicValuation(0.6)
        }
        self.utilities = {
            "agent_0": LinearUtility(),
            "agent_1": LinearUtility()
        }
        self.action_spaces = {
            "agent_0": self.action_space,
            "agent_1": self.action_space
        }
        self.observation_spaces = {
            "agent_0": self.obs_space,
            "agent_1": self.obs_space
        }
        
        self.env = AuctionEnv(
            valuations=self.valuations,
            utilities=self.utilities,
            action_spaces=self.action_spaces,
            observation_spaces=self.observation_spaces,
            max_steps=3
        )
        
        # Reset environment
        self.env.reset()
    
    def test_step_basic_functionality(self):
        """Test basic step functionality."""
        action_dict = {
            "agent_0": np.array([0.9]),
            "agent_1": np.array([0.7])
        }
        
        obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict = self.env.step(action_dict)
        
        # Check return format
        assert set(obs_dict.keys()) == {"agent_0", "agent_1"}
        assert set(rewards_dict.keys()) == {"agent_0", "agent_1"}
        assert set(terminated_dict.keys()) == {"agent_0", "agent_1", "__all__"}
        assert set(truncated_dict.keys()) == {"agent_0", "agent_1", "__all__"}
        assert set(info_dict.keys()) == {"step", "max_steps", "n_agents", "mechanism_type","allocations","payments","bids","valuations"}
        
        # Check step increment
        assert self.env.current_step == 1
        
        # Check episode data storage
        assert len(self.env.episode_data["bids"]) == 1
        assert len(self.env.episode_data["valuations"]) == 1
        assert len(self.env.episode_data["allocations"]) == 1
        assert len(self.env.episode_data["payments"]) == 1
    
    def test_step_missing_action_raises_error(self):
        """Test that step fails when action is missing for an agent."""
        action_dict = {
            "agent_0": np.array([0.9])
            # Missing agent_1
        }
        
        with pytest.raises(KeyError):
            self.env.step(action_dict)
    
    def test_step_episode_truncation(self):
        """Test that episode gets truncated after max_steps."""
        action_dict = {
            "agent_0": np.array([0.9]),
            "agent_1": np.array([0.7])
        }
        
        # Run until max_steps
        for _ in range(3):
            obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict = self.env.step(action_dict)
        
        # Check truncation
        assert truncated_dict["__all__"] == True
        assert self.env.current_step == 3
    
    def test_step_termination_always_false(self):
        """Test that termination is always False (episodes only truncate)."""
        action_dict = {
            "agent_0": np.array([0.9]),
            "agent_1": np.array([0.7])
        }
        
        obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict = self.env.step(action_dict)
        
        assert terminated_dict["__all__"] == False
        assert all(terminated == False for terminated in terminated_dict.values() if terminated != "__all__")


class TestAuctionEnvRewards:
    """Test reward computation in AuctionEnv."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.agent_keys = ["agent_0", "agent_1"]
        self.valuations = {
            "agent_0": DeterministicValuation(0.8),
            "agent_1": DeterministicValuation(0.6)
        }
        self.utilities = {
            "agent_0": LinearUtility(),
            "agent_1": RiskAverseUtility()  # Different utility for testing
        }
        self.action_spaces = {
            "agent_0": self.action_space,
            "agent_1": self.action_space
        }
        self.observation_spaces = {
            "agent_0": self.obs_space,
            "agent_1": self.obs_space
        }
        
        self.env = AuctionEnv(
            valuations=self.valuations,
            utilities=self.utilities,
            action_spaces=self.action_spaces,
            observation_spaces=self.observation_spaces
        )
        
        self.env.reset()
    
    def test_reward_computation_winner(self):
        """Test reward computation when agent wins."""
        # Mock mechanism to return known allocations and payments
        with patch.object(self.env.mechanism, 'run_auction') as mock_auction:
            mock_auction.return_value = (
                np.array([True, False]),  # agent_0 wins
                np.array([0.7, 0.0])     # agent_0 pays 0.7, agent_1 pays 0
            )
            
            action_dict = {
                "agent_0": np.array([0.9]),
                "agent_1": np.array([0.7])
            }
            
            obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict = self.env.step(action_dict)
            
            # agent_0 should get reward valuation - payment = 0.8 - 0.7 = 0.1
            # agent_1 should get 0 reward (didn't win)
            assert np.isclose(rewards_dict["agent_0"], 0.1)
            assert np.isclose(rewards_dict["agent_1"], 0.0)
    
    def test_reward_computation_loser(self):
        """Test reward computation when agent loses."""
        with patch.object(self.env.mechanism, 'run_auction') as mock_auction:
            mock_auction.return_value = (
                np.array([False, True]),  # agent_1 wins
                np.array([0.0, 0.9])     # agent_0 pays 0, agent_1 pays 0.9
            )
            
            action_dict = {
                "agent_0": np.array([0.7]),
                "agent_1": np.array([0.9])
            }
            
            obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict = self.env.step(action_dict)
            
            # agent_0 should get 0 reward (didn't win)
            # agent_1 should get negative reward with risk averse utility
            # RiskAverseUtility: log(1 + valuation - payment) = log(1 + 0.6 - 0.9) = log(0.7)
            assert np.isclose(rewards_dict["agent_0"], 0.0)
            assert np.isclose(rewards_dict["agent_1"], np.log(0.7))


class TestAuctionEnvEpisodeSummary:
    """Test episode summary functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.agent_keys = ["agent_0", "agent_1"]
        self.valuations = {
            "agent_0": DeterministicValuation(0.8),
            "agent_1": DeterministicValuation(0.6)
        }
        self.utilities = {
            "agent_0": LinearUtility(),
            "agent_1": LinearUtility()
        }
        self.action_spaces = {
            "agent_0": self.action_space,
            "agent_1": self.action_space
        }
        self.observation_spaces = {
            "agent_0": self.obs_space,
            "agent_1": self.obs_space
        }
        
        self.env = AuctionEnv(
            valuations=self.valuations,
            utilities=self.utilities,
            action_spaces=self.action_spaces,
            observation_spaces=self.observation_spaces
        )
    
    def test_empty_episode_summary(self):
        """Test episode summary when no steps have been taken."""
        summary = self.env.get_episode_summary()
        
        assert "error" in summary
        assert summary["error"] == "No episode data available"
    
    def test_episode_summary_with_data(self):
        """Test episode summary with actual episode data."""
        self.env.reset()
        
        action_dict = {
            "agent_0": np.array([0.9]),
            "agent_1": np.array([0.7])
        }
        
        # Run a few steps
        for _ in range(3):
            self.env.step(action_dict)
        
        summary = self.env.get_episode_summary()
        
        # Check summary structure
        assert "episode_length" in summary
        assert "total_revenue" in summary
        assert "average_bid" in summary
        assert "winner_distribution" in summary
        assert "agent_performance" in summary
        assert "mechanism_type" in summary
        
        # Check values
        assert summary["episode_length"] == 3
        assert summary["mechanism_type"] == "second_price"
        assert set(summary["winner_distribution"].keys()) == {"agent_0", "agent_1"}
        assert set(summary["agent_performance"].keys()) == {"agent_0", "agent_1"}
        
        # Check agent performance structure
        for agent_key in ["agent_0", "agent_1"]:
            agent_perf = summary["agent_performance"][agent_key]
            assert "wins" in agent_perf
            assert "total_payments" in agent_perf
            assert "total_valuations" in agent_perf
            assert "win_rate" in agent_perf


class TestAuctionEnvWinnerDistribution:
    """Test winner distribution calculation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.agent_keys = ["agent_0", "agent_1"]
        self.valuations = {
            "agent_0": DeterministicValuation(0.8),
            "agent_1": DeterministicValuation(0.6)
        }
        self.utilities = {
            "agent_0": LinearUtility(),
            "agent_1": LinearUtility()
        }
        self.action_spaces = {
            "agent_0": self.action_space,
            "agent_1": self.action_space
        }
        self.observation_spaces = {
            "agent_0": self.obs_space,
            "agent_1": self.obs_space
        }
        
        self.env = AuctionEnv(
            valuations=self.valuations,
            utilities=self.utilities,
            action_spaces=self.action_spaces,
            observation_spaces=self.observation_spaces
        )
    
    def test_winner_distribution_empty(self):
        """Test winner distribution with no episode data."""
        distribution = self.env._get_winner_distribution()
        
        assert distribution == {"agent_0": 0, "agent_1": 0}
    
    def test_winner_distribution_with_data(self):
        """Test winner distribution with episode data."""
        # Manually set episode data
        self.env.episode_data["allocations"] = [
            np.array([True, False]),   # agent_0 wins
            np.array([False, True]),   # agent_1 wins
            np.array([True, False]),   # agent_0 wins
        ]
        
        distribution = self.env._get_winner_distribution()
        
        assert distribution["agent_0"] == 2
        assert distribution["agent_1"] == 1


class TestAuctionEnvRender:
    """Test rendering functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.agent_keys = ["agent_0", "agent_1"]
        self.valuations = {
            "agent_0": DeterministicValuation(0.8),
            "agent_1": DeterministicValuation(0.6)
        }
        self.utilities = {
            "agent_0": LinearUtility(),
            "agent_1": LinearUtility()
        }
        self.action_spaces = {
            "agent_0": self.action_space,
            "agent_1": self.action_space
        }
        self.observation_spaces = {
            "agent_0": self.obs_space,
            "agent_1": self.obs_space
        }
        
        self.env = AuctionEnv(
            valuations=self.valuations,
            utilities=self.utilities,
            action_spaces=self.action_spaces,
            observation_spaces=self.observation_spaces
        )
    
    def test_render_human_mode(self):
        """Test rendering in human mode."""
        # Should not raise an error
        self.env.render(mode="human")
    
    def test_render_invalid_mode(self):
        """Test that render raises error for invalid mode."""
        with pytest.raises(ValueError, match="Unsupported render mode"):
            self.env.render(mode="invalid_mode")
    
    def test_render_with_episode_data(self):
        """Test rendering when episode has data."""
        self.env.reset()
        
        action_dict = {
            "agent_0": np.array([0.9]),
            "agent_1": np.array([0.7])
        }
        
        self.env.step(action_dict)
        
        # Should not raise an error
        self.env.render(mode="human")


class TestAuctionEnvIntegration:
    """Integration tests for AuctionEnv."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.agent_keys = ["agent_0", "agent_1", "agent_2"]
        self.valuations = {
            "agent_0": DeterministicValuation(0.8),
            "agent_1": DeterministicValuation(0.6),
            "agent_2": DeterministicValuation(0.9)
        }
        self.utilities = {
            "agent_0": LinearUtility(),
            "agent_1": LinearUtility(),
            "agent_2": RiskAverseUtility()
        }
        self.action_spaces = {
            "agent_0": self.action_space,
            "agent_1": self.action_space,
            "agent_2": self.action_space
        }
        self.observation_spaces = {
            "agent_0": self.obs_space,
            "agent_1": self.obs_space,
            "agent_2": self.obs_space
        }
        
        self.env = AuctionEnv(
            valuations=self.valuations,
            utilities=self.utilities,
            action_spaces=self.action_spaces,
            observation_spaces=self.observation_spaces,
            max_steps=5
        )
    
    def test_full_episode_simulation(self):
        """Test a full episode simulation."""
        obs_dict, info_dict = self.env.reset()
        
        action_dict = {
            "agent_0": np.array([0.8]),
            "agent_1": np.array([0.6]),
            "agent_2": np.array([0.9])
        }
        
        # Run full episode
        for step in range(5):
            obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict = self.env.step(action_dict)
            
            # Check that all agents are present in returns
            assert set(obs_dict.keys()) == {"agent_0", "agent_1", "agent_2"}
            assert set(rewards_dict.keys()) == {"agent_0", "agent_1", "agent_2"}
            assert set(terminated_dict.keys()) == {"agent_0", "agent_1", "agent_2", "__all__"}
            assert set(truncated_dict.keys()) == {"agent_0", "agent_1", "agent_2", "__all__"}
            assert set(info_dict.keys()) == {"step", "max_steps", "n_agents", "mechanism_type","allocations","payments","bids","valuations"}
            
            # Check step progression
            assert self.env.current_step == step + 1
        
        # Check final state
        assert self.env.current_step == 5
        assert truncated_dict["__all__"] == True
        
        # Check episode summary
        summary = self.env.get_episode_summary()
        assert summary["episode_length"] == 5
        assert summary["mechanism_type"] == "second_price" 
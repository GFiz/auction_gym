import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from gymnasium import spaces

from auction_gym.core.policy import LinearBidder


class TestLinearBidder:
    """Test the LinearBidder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    def test_initialization(self):
        """Test that LinearBidder initializes correctly."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        assert hasattr(bidder, 'theta')
        assert isinstance(bidder.theta, torch.nn.Parameter)
        assert bidder.theta.item() == 1.0
        assert bidder.theta.dtype == torch.float32
    
    def test_initialization_with_kwargs(self):
        """Test that LinearBidder initializes correctly with additional kwargs."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space,
            inference_only=True,
            model_config={'test_param': 'test_value'}
        )
        
        assert hasattr(bidder, 'theta')
        assert isinstance(bidder.theta, torch.nn.Parameter)
        assert bidder.theta.item() == 1.0
    
    def test_setup_method(self):
        """Test the setup method."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        # The setup method should have been called during initialization
        assert hasattr(bidder, 'theta')
        assert bidder.theta.item() == 1.0
        
        # Call setup explicitly to test it
        bidder.setup()
        assert bidder.theta.item() == 1.0
    
    def test_forward_with_tensor_observation(self):
        """Test forward pass with tensor observation."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        # Create test observation tensor
        observation = torch.tensor([[0.5]], dtype=torch.float32)
        batch = {"obs": observation}
        
        result = bidder._forward(batch)
        
        # Check that the result contains expected keys
        assert "action" in result
        assert "action_dist_inputs" in result
        
        # Check that both action and action_dist_inputs are the same
        assert torch.equal(result["action"], result["action_dist_inputs"])
        
        # Check that the bid is computed correctly: observation * theta
        expected_bid = observation * bidder.theta
        expected_bid = torch.clamp(expected_bid, 0.0, 1.0)
        assert torch.equal(result["action"], expected_bid)
    
    def test_forward_with_numpy_observation(self):
        """Test forward pass with numpy observation."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        # Create test observation as numpy array
        observation = np.array([[0.3]], dtype=np.float32)
        batch = {"obs": observation}
        
        result = bidder._forward(batch)
        
        # Check that the result contains expected keys
        assert "action" in result
        assert "action_dist_inputs" in result
        
        # Check that the observation was converted to tensor
        expected_bid = torch.tensor(observation, dtype=torch.float32) * bidder.theta
        expected_bid = torch.clamp(expected_bid, 0.0, 1.0)
        assert torch.equal(result["action"], expected_bid)
    
    def test_forward_with_1d_observation(self):
        """Test forward pass with 1D observation (should add batch dimension)."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        # Create 1D observation
        observation = torch.tensor([0.7], dtype=torch.float32)
        batch = {"obs": observation}
        
        result = bidder._forward(batch)
        
        # Check that the result contains expected keys
        assert "action" in result
        assert "action_dist_inputs" in result
        
        # Check that the observation was reshaped to 2D
        expected_bid = observation.unsqueeze(0) * bidder.theta
        expected_bid = torch.clamp(expected_bid, 0.0, 1.0)
        assert torch.equal(result["action"], expected_bid)
    
    def test_forward_with_observations_key(self):
        """Test forward pass when observation is under 'observations' key."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        # Create test observation
        observation = torch.tensor([[0.4]], dtype=torch.float32)
        batch = {"observations": observation}
        
        result = bidder._forward(batch)
        
        # Check that the result contains expected keys
        assert "action" in result
        assert "action_dist_inputs" in result
        
        # Check that the bid is computed correctly
        expected_bid = observation * bidder.theta
        expected_bid = torch.clamp(expected_bid, 0.0, 1.0)
        assert torch.equal(result["action"], expected_bid)
    
    def test_forward_with_missing_observation(self):
        """Test forward pass when no observation is provided."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        batch = {"other_key": torch.tensor([[0.5]])}
        
        # Should raise ValueError when no observation is provided
        with pytest.raises(ValueError, match="No observation provided in batch"):
            bidder._forward(batch)
    
    def test_forward_clamping_behavior(self):
        """Test that bids are properly clamped to [0, 1] range."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        # Test with observation that would result in bid > 1
        observation = torch.tensor([[1.5]], dtype=torch.float32)
        batch = {"obs": observation}
        
        result = bidder._forward(batch)
        
        # Bid should be clamped to 1.0
        assert torch.allclose(result["action"], torch.tensor([[1.0]]))
        
        # Test with observation that would result in bid < 0
        observation = torch.tensor([[-0.5]], dtype=torch.float32)
        batch = {"obs": observation}
        
        result = bidder._forward(batch)
        
        # Bid should be clamped to 0.0
        assert torch.allclose(result["action"], torch.tensor([[0.0]]))
    
    def test_forward_with_different_theta_values(self):
        """Test forward pass with different theta parameter values."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        # Set theta to a different value
        bidder.theta.data = torch.tensor(0.5, dtype=torch.float32)
        
        observation = torch.tensor([[0.8]], dtype=torch.float32)
        batch = {"obs": observation}
        
        result = bidder._forward(batch)
        
        # Check that the bid is computed correctly with new theta
        expected_bid = observation * 0.5
        expected_bid = torch.clamp(expected_bid, 0.0, 1.0)
        assert torch.equal(result["action"], expected_bid)
    
    def test_forward_with_batch_size_greater_than_one(self):
        """Test forward pass with batch size greater than one."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        # Create batch with multiple observations
        observation = torch.tensor([[0.3], [0.7], [0.9]], dtype=torch.float32)
        batch = {"obs": observation}
        
        result = bidder._forward(batch)
        
        # Check that the result has the correct batch size
        assert result["action"].shape == (3, 1)
        assert result["action_dist_inputs"].shape == (3, 1)
        
        # Check that each bid is computed correctly
        expected_bids = observation * bidder.theta
        expected_bids = torch.clamp(expected_bids, 0.0, 1.0)
        assert torch.equal(result["action"], expected_bids)
    
    def test_forward_with_additional_kwargs(self):
        """Test forward pass with additional keyword arguments."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        observation = torch.tensor([[0.6]], dtype=torch.float32)
        batch = {"obs": observation}
        
        # Pass additional kwargs
        result = bidder._forward(batch, extra_arg="test", another_arg=123)
        
        # Should handle additional kwargs gracefully
        assert "action" in result
        assert "action_dist_inputs" in result
    
    def test_theta_parameter_gradients(self):
        """Test that theta parameter can accumulate gradients."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        observation = torch.tensor([[0.5]], dtype=torch.float32)
        batch = {"obs": observation}
        
        # Perform forward pass
        result = bidder._forward(batch)
        
        # Check that theta requires gradients
        assert bidder.theta.requires_grad
        
        # Perform backward pass
        loss = result["action"].sum()
        loss.backward()
        
        # Check that gradients are computed
        assert bidder.theta.grad is not None
        assert bidder.theta.grad.shape == bidder.theta.shape


class TestLinearBidderEdgeCases:
    """Test edge cases and error conditions for LinearBidder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    def test_forward_with_empty_batch(self):
        """Test forward pass with empty batch."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        batch = {}
        
        # Should raise ValueError when no observation is provided
        with pytest.raises(ValueError, match="No observation provided in batch"):
            bidder._forward(batch)
    
    def test_forward_with_none_observation(self):
        """Test forward pass with None observation."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        batch = {"obs": None}
        
        # Should raise ValueError when observation is None
        with pytest.raises(ValueError, match="No observation provided in batch"):
            bidder._forward(batch)
    
    def test_forward_with_mixed_data_types(self):
        """Test forward pass with mixed data types in batch."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        # Create batch with mixed data types
        batch = {
            "obs": torch.tensor([[0.5]]),
            "other_data": "string",
            "numeric_data": 42,
            "list_data": [1, 2, 3]
        }
        
        result = bidder._forward(batch)
        
        # Should handle mixed data types gracefully
        assert "action" in result
        assert "action_dist_inputs" in result


class TestLinearBidderIntegration:
    """Integration tests for LinearBidder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    def test_multiple_forward_passes(self):
        """Test multiple forward passes with the same bidder."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        # Perform multiple forward passes
        observations = [
            torch.tensor([[0.2]]),
            torch.tensor([[0.5]]),
            torch.tensor([[0.8]])
        ]
        
        results = []
        for obs in observations:
            batch = {"obs": obs}
            result = bidder._forward(batch)
            results.append(result)
        
        # Check that all results are consistent
        for i, result in enumerate(results):
            assert "action" in result
            assert "action_dist_inputs" in result
            
            expected_bid = observations[i] * bidder.theta
            expected_bid = torch.clamp(expected_bid, 0.0, 1.0)
            assert torch.equal(result["action"], expected_bid)
    
    def test_parameter_updates(self):
        """Test that theta parameter can be updated and affects subsequent forward passes."""
        bidder = LinearBidder(
            observation_space=self.obs_space,
            action_space=self.action_space
        )
        
        # Initial forward pass
        observation = torch.tensor([[0.5]])
        batch = {"obs": observation}
        initial_result = bidder._forward(batch)
        
        # Update theta parameter
        bidder.theta.data = torch.tensor(0.3, dtype=torch.float32)
        
        # Second forward pass
        updated_result = bidder._forward(batch)
        
        # Results should be different due to theta change
        assert not torch.equal(initial_result["action"], updated_result["action"])
        
        # Check that updated result is correct
        expected_bid = observation * 0.3
        expected_bid = torch.clamp(expected_bid, 0.0, 1.0)
        assert torch.equal(updated_result["action"], expected_bid)


@pytest.mark.parametrize("observation_value,expected_bid", [
    (0.0, 0.0),
    (0.5, 0.5),
    (1.0, 1.0),
    (1.5, 1.0),  # Should be clamped
    (-0.5, 0.0),  # Should be clamped
])
def test_forward_with_different_observation_values(observation_value, expected_bid):
    """Test forward pass with different observation values."""
    obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    bidder = LinearBidder(
        observation_space=obs_space,
        action_space=action_space
    )
    
    observation = torch.tensor([[observation_value]], dtype=torch.float32)
    batch = {"obs": observation}
    
    result = bidder._forward(batch)
    
    assert torch.allclose(result["action"], torch.tensor([[expected_bid]])) 
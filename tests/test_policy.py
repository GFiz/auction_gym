import pytest
import torch
import numpy as np
from gymnasium import spaces
from unittest.mock import Mock, patch, MagicMock

from auction_gym.core.policy import LinearBidder
from ray.rllib.core.columns import Columns


class TestLinearBidderInitialization:
    """Test LinearBidder initialization and setup."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    def test_valid_initialization(self):
        """Test that LinearBidder initializes correctly with valid parameters."""
        bidder = LinearBidder(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        
        assert bidder.observation_space == self.observation_space
        assert bidder.action_space == self.action_space
    
    def test_initialization_with_kwargs(self):
        """Test initialization with additional keyword arguments."""
        bidder = LinearBidder(
            observation_space=self.observation_space,
            action_space=self.action_space,
            inference_only=True,
            model_config={"test": "config"},
            catalog_class=Mock()
        )
        
        assert bidder.observation_space == self.observation_space
        assert bidder.action_space == self.action_space
    
    def test_setup_method(self):
        """Test that setup method initializes parameters correctly."""
        bidder = LinearBidder(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        
        # Call setup manually since it's normally called by parent
        bidder.setup()
        
        # Check that parameters are created and have correct values
        assert hasattr(bidder, 'mu')
        assert hasattr(bidder, 'sigma')
        assert isinstance(bidder.mu, torch.nn.Parameter)
        assert isinstance(bidder.sigma, torch.nn.Parameter)
        assert bidder.mu.dtype == torch.float32
        assert bidder.sigma.dtype == torch.float32
        assert torch.equal(bidder.mu, torch.tensor(1.0, dtype=torch.float32))
        assert torch.equal(bidder.sigma, torch.tensor(1.0, dtype=torch.float32))


class TestLinearBidderForward:
    """Test LinearBidder forward pass functionality for training (Gaussian Diagonal)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.bidder = LinearBidder(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        self.bidder.setup()
    
    def test_forward_basic_functionality(self):
        """Test basic forward pass functionality."""
        # Create test batch with observations
        batch_size = 3
        observations = torch.randn(batch_size, 1, dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward(batch)
        
        # Check return format
        assert Columns.ACTION_DIST_INPUTS in result
        action_params = result[Columns.ACTION_DIST_INPUTS]
        
        # Check output shape: should be (batch_size, 2) for mean and log_std
        assert action_params.shape == (batch_size, 2)
        assert action_params.dtype == torch.float32
    
    def test_forward_single_observation(self):
        """Test forward pass with single observation."""
        observations = torch.tensor([[0.5]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward(batch)
        action_params = result[Columns.ACTION_DIST_INPUTS]
        
        assert action_params.shape == (1, 2)
        
        # Check that the computation is correct
        # First column should be mean_logits = mu * observation
        # Second column should be log_std = log(sigma)
        expected_mean = self.bidder.mu * observations[0, 0]
        expected_log_std = torch.log(self.bidder.sigma)
        
        assert torch.isclose(action_params[0, 0], expected_mean)
        assert torch.isclose(action_params[0, 1], expected_log_std)
    
    def test_forward_multiple_observations(self):
        """Test forward pass with multiple observations."""
        observations = torch.tensor([[0.3], [0.7], [0.1]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward(batch)
        action_params = result[Columns.ACTION_DIST_INPUTS]
        
        assert action_params.shape == (3, 2)
        
        # Check that each row has correct computation
        for i in range(3):
            expected_mean = self.bidder.mu * observations[i, 0]
            expected_log_std = torch.log(self.bidder.sigma)
            
            assert torch.isclose(action_params[i, 0], expected_mean)
            assert torch.isclose(action_params[i, 1], expected_log_std)
    
    def test_forward_with_additional_kwargs(self):
        """Test forward pass with additional keyword arguments."""
        observations = torch.tensor([[0.5]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        # Should not raise an error
        result = self.bidder._forward(batch, extra_arg="test")
        assert Columns.ACTION_DIST_INPUTS in result
    
    def test_forward_parameter_gradients(self):
        """Test that forward pass maintains gradient information."""
        observations = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward(batch)
        action_params = result[Columns.ACTION_DIST_INPUTS]
        
        # Check that gradients can be computed
        loss = action_params.sum()
        loss.backward()
        
        # Check that parameters have gradients
        assert self.bidder.mu.grad is not None
        assert self.bidder.sigma.grad is not None


class TestLinearBidderForwardInference:
    """Test LinearBidder forward inference functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.bidder = LinearBidder(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        self.bidder.setup()
    
    def test_forward_inference_basic_functionality(self):
        """Test basic forward inference functionality."""
        # Create test batch with observations
        batch_size = 3
        observations = torch.randn(batch_size, 1, dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward_inference(batch)
        
        # Check return format
        assert Columns.ACTIONS in result
        actions = result[Columns.ACTIONS]
        
        # Check output shape: should be (batch_size, 1) for direct actions
        assert actions.shape == (batch_size, 1)
        assert actions.dtype == torch.float32
    
    def test_forward_inference_single_observation(self):
        """Test forward inference with single observation."""
        observations = torch.tensor([[0.5]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward_inference(batch)
        actions = result[Columns.ACTIONS]
        
        assert actions.shape == (1, 1)
        
        # Check that the computation is correct
        # Should be mu * observation
        expected_action = self.bidder.mu * observations[0, 0]
        assert torch.isclose(actions[0, 0], expected_action)
    
    def test_forward_inference_multiple_observations(self):
        """Test forward inference with multiple observations."""
        observations = torch.tensor([[0.3], [0.7], [0.1]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward_inference(batch)
        actions = result[Columns.ACTIONS]
        
        assert actions.shape == (3, 1)
        
        # Check that each row has correct computation
        for i in range(3):
            expected_action = self.bidder.mu * observations[i, 0]
            assert torch.isclose(actions[i, 0], expected_action)
    
    def test_forward_inference_with_additional_kwargs(self):
        """Test forward inference with additional keyword arguments."""
        observations = torch.tensor([[0.5]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        # Should not raise an error
        result = self.bidder._forward_inference(batch, extra_arg="test")
        assert Columns.ACTIONS in result
    
    def test_forward_inference_parameter_gradients(self):
        """Test that forward inference maintains gradient information."""
        observations = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward_inference(batch)
        actions = result[Columns.ACTIONS]
        
        # Check that gradients can be computed
        loss = actions.sum()
        loss.backward()
        
        # Check that parameters have gradients
        assert self.bidder.mu.grad is not None
        # Note: sigma is not used in inference, so it won't have gradients


class TestLinearBidderParameterUpdates:
    """Test LinearBidder parameter update functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.bidder = LinearBidder(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        self.bidder.setup()
    
    def test_parameter_initialization(self):
        """Test that parameters are initialized with correct values."""
        assert torch.equal(self.bidder.mu, torch.tensor(1.0, dtype=torch.float32))
        assert torch.equal(self.bidder.sigma, torch.tensor(1.0, dtype=torch.float32))
    
    def test_parameter_modification(self):
        """Test that parameters can be modified."""
        # Modify parameters
        with torch.no_grad():
            self.bidder.mu.data.fill_(2.0)
            self.bidder.sigma.data.fill_(0.5)
        
        assert torch.equal(self.bidder.mu, torch.tensor(2.0, dtype=torch.float32))
        assert torch.equal(self.bidder.sigma, torch.tensor(0.5, dtype=torch.float32))
    
    def test_parameter_gradient_flow_training(self):
        """Test that gradients flow through parameters correctly during training."""
        observations = torch.tensor([[0.5]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward(batch)
        action_params = result[Columns.ACTION_DIST_INPUTS]
        
        # Compute loss and backward pass
        loss = action_params.sum()
        loss.backward()
        
        # Check gradients
        assert self.bidder.mu.grad is not None
        assert self.bidder.sigma.grad is not None
        
        # mu gradient should be observation value
        assert torch.isclose(self.bidder.mu.grad, torch.tensor(0.5))
        # sigma gradient should be from log_std computation
        assert self.bidder.sigma.grad is not None
    
    def test_parameter_gradient_flow_inference(self):
        """Test that gradients flow through parameters correctly during inference."""
        observations = torch.tensor([[0.5]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward_inference(batch)
        actions = result[Columns.ACTIONS]
        
        # Compute loss and backward pass
        loss = actions.sum()
        loss.backward()
        
        # Check gradients
        assert self.bidder.mu.grad is not None
        # sigma is not used in inference, so it won't have gradients
        
        # mu gradient should be observation value
        assert torch.isclose(self.bidder.mu.grad, torch.tensor(0.5))


class TestLinearBidderEdgeCases:
    """Test LinearBidder edge cases and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.bidder = LinearBidder(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        self.bidder.setup()
    
    def test_forward_zero_observations(self):
        """Test forward pass with zero observations."""
        observations = torch.tensor([[0.0]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward(batch)
        action_params = result[Columns.ACTION_DIST_INPUTS]
        
        # mean should be 0, log_std should be log(1) = 0
        assert torch.isclose(action_params[0, 0], torch.tensor(0.0))
        assert torch.isclose(action_params[0, 1], torch.tensor(0.0))
    
    def test_forward_inference_zero_observations(self):
        """Test forward inference with zero observations."""
        observations = torch.tensor([[0.0]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward_inference(batch)
        actions = result[Columns.ACTIONS]
        
        # action should be 0
        assert torch.isclose(actions[0, 0], torch.tensor(0.0))
    
    def test_forward_negative_observations(self):
        """Test forward pass with negative observations."""
        observations = torch.tensor([[-0.5]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward(batch)
        action_params = result[Columns.ACTION_DIST_INPUTS]
        
        # mean should be -0.5, log_std should be 0
        assert torch.isclose(action_params[0, 0], torch.tensor(-0.5))
        assert torch.isclose(action_params[0, 1], torch.tensor(0.0))
    
    def test_forward_inference_negative_observations(self):
        """Test forward inference with negative observations."""
        observations = torch.tensor([[-0.5]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward_inference(batch)
        actions = result[Columns.ACTIONS]
        
        # action should be -0.5
        assert torch.isclose(actions[0, 0], torch.tensor(-0.5))
    
    def test_forward_large_observations(self):
        """Test forward pass with large observation values."""
        observations = torch.tensor([[1000.0]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward(batch)
        action_params = result[Columns.ACTION_DIST_INPUTS]
        
        # mean should be 1000, log_std should be 0
        assert torch.isclose(action_params[0, 0], torch.tensor(1000.0))
        assert torch.isclose(action_params[0, 1], torch.tensor(0.0))
    
    def test_forward_inference_large_observations(self):
        """Test forward inference with large observation values."""
        observations = torch.tensor([[1000.0]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward_inference(batch)
        actions = result[Columns.ACTIONS]
        
        # action should be 1000
        assert torch.isclose(actions[0, 0], torch.tensor(1000.0))
    
    def test_forward_missing_observation_key(self):
        """Test forward pass with missing observation key."""
        batch = {"wrong_key": torch.tensor([[0.5]], dtype=torch.float32)}
        
        with pytest.raises(KeyError):
            self.bidder._forward(batch)
    
    def test_forward_inference_missing_observation_key(self):
        """Test forward inference with missing observation key."""
        batch = {"wrong_key": torch.tensor([[0.5]], dtype=torch.float32)}
        
        with pytest.raises(KeyError):
            self.bidder._forward_inference(batch)
    
    def test_forward_empty_batch(self):
        """Test forward pass with empty batch."""
        batch = {}
        
        with pytest.raises(KeyError):
            self.bidder._forward(batch)
    
    def test_forward_inference_empty_batch(self):
        """Test forward inference with empty batch."""
        batch = {}
        
        with pytest.raises(KeyError):
            self.bidder._forward_inference(batch)


class TestLinearBidderIntegration:
    """Integration tests for LinearBidder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.bidder = LinearBidder(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        self.bidder.setup()
    
    def test_full_training_step(self):
        """Test a full training step with parameter updates."""
        # Create optimizer
        optimizer = torch.optim.SGD([self.bidder.mu, self.bidder.sigma], lr=0.01)
        
        # Initial parameter values
        initial_mu = self.bidder.mu.clone()
        initial_sigma = self.bidder.sigma.clone()
        
        # Forward pass
        observations = torch.tensor([[0.5]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        result = self.bidder._forward(batch)
        action_params = result[Columns.ACTION_DIST_INPUTS]
        
        # Compute loss
        target = torch.tensor([[0.8, 0.3]], dtype=torch.float32)
        loss = torch.nn.functional.mse_loss(action_params, target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters were updated
        assert not torch.equal(self.bidder.mu, initial_mu)
        assert not torch.equal(self.bidder.sigma, initial_sigma)
    
    def test_batch_processing_training(self):
        """Test processing multiple batches for training."""
        batch_sizes = [1, 5, 10]
        
        for batch_size in batch_sizes:
            observations = torch.randn(batch_size, 1, dtype=torch.float32)
            batch = {Columns.OBS: observations}
            
            result = self.bidder._forward(batch)
            action_params = result[Columns.ACTION_DIST_INPUTS]
            
            assert action_params.shape == (batch_size, 2)
    
    def test_batch_processing_inference(self):
        """Test processing multiple batches for inference."""
        batch_sizes = [1, 5, 10]
        
        for batch_size in batch_sizes:
            observations = torch.randn(batch_size, 1, dtype=torch.float32)
            batch = {Columns.OBS: observations}
            
            result = self.bidder._forward_inference(batch)
            actions = result[Columns.ACTIONS]
            
            assert actions.shape == (batch_size, 1)
    
    def test_parameter_persistence(self):
        """Test that parameters persist across multiple forward passes."""
        # Modify parameters
        with torch.no_grad():
            self.bidder.mu.data.fill_(2.0)
            self.bidder.sigma.data.fill_(0.5)
        
        # Multiple forward passes for training
        for _ in range(5):
            observations = torch.tensor([[0.5]], dtype=torch.float32)
            batch = {Columns.OBS: observations}
            result = self.bidder._forward(batch)
            action_params = result[Columns.ACTION_DIST_INPUTS]
            
            # Check that parameters remain the same
            assert torch.isclose(action_params[0, 0], torch.tensor(1.0))  # 2.0 * 0.5
            # Check log_std with more robust comparison
            expected_log_std = torch.log(torch.tensor(0.5))
            assert torch.isclose(action_params[0, 1], expected_log_std, atol=1e-6)
        
        # Multiple forward passes for inference
        for _ in range(5):
            observations = torch.tensor([[0.5]], dtype=torch.float32)
            batch = {Columns.OBS: observations}
            result = self.bidder._forward_inference(batch)
            actions = result[Columns.ACTIONS]
            
            # Check that parameters remain the same
            assert torch.isclose(actions[0, 0], torch.tensor(1.0))  # 2.0 * 0.5


class TestLinearBidderCompatibility:
    """Test LinearBidder compatibility with RLlib framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    def test_rllib_module_compatibility(self):
        """Test that LinearBidder is compatible with RLlib module interface."""
        bidder = LinearBidder(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        
        # Check that it inherits from TorchRLModule
        from ray.rllib.core.rl_module.torch import TorchRLModule
        assert isinstance(bidder, TorchRLModule)
    
    def test_columns_compatibility_training(self):
        """Test that LinearBidder uses correct RLlib column names for training."""
        bidder = LinearBidder(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        bidder.setup()
        
        # Test that it uses the correct column names for training
        observations = torch.tensor([[0.5]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = bidder._forward(batch)
        
        # Should return action distribution inputs
        assert Columns.ACTION_DIST_INPUTS in result
    
    def test_columns_compatibility_inference(self):
        """Test that LinearBidder uses correct RLlib column names for inference."""
        bidder = LinearBidder(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        bidder.setup()
        
        # Test that it uses the correct column names for inference
        observations = torch.tensor([[0.5]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = bidder._forward_inference(batch)
        
        # Should return actions
        assert Columns.ACTIONS in result
    
    def test_inference_only_mode(self):
        """Test LinearBidder in inference-only mode."""
        bidder = LinearBidder(
            observation_space=self.observation_space,
            action_space=self.action_space,
            inference_only=True
        )
        bidder.setup()
        
        # Should work for both training and inference
        observations = torch.tensor([[0.5]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        # Training forward pass
        result = bidder._forward(batch)
        assert Columns.ACTION_DIST_INPUTS in result
        
        # Inference forward pass
        result = bidder._forward_inference(batch)  # Changed from self.bidder to bidder
        assert Columns.ACTIONS in result


class TestLinearBidderPerformance:
    """Performance tests for LinearBidder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.bidder = LinearBidder(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        self.bidder.setup()
    
    def test_large_batch_performance_training(self):
        """Test performance with large batch sizes for training."""
        batch_size = 1000
        observations = torch.randn(batch_size, 1, dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        # Time the forward pass
        import time
        start_time = time.time()
        
        result = self.bidder._forward(batch)
        action_params = result[Columns.ACTION_DIST_INPUTS]
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete in reasonable time (less than 1 second)
        assert processing_time < 1.0
        assert action_params.shape == (batch_size, 2)
    
    def test_large_batch_performance_inference(self):
        """Test performance with large batch sizes for inference."""
        batch_size = 1000
        observations = torch.randn(batch_size, 1, dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        # Time the forward pass
        import time
        start_time = time.time()
        
        result = self.bidder._forward_inference(batch)
        actions = result[Columns.ACTIONS]
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete in reasonable time (less than 1 second)
        assert processing_time < 1.0
        assert actions.shape == (batch_size, 1)
    
    @pytest.mark.parametrize("batch_size", [1, 10, 100, 1000])
    def test_batch_size_scaling_training(self, batch_size):
        """Test that LinearBidder scales with different batch sizes for training."""
        observations = torch.randn(batch_size, 1, dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward(batch)
        action_params = result[Columns.ACTION_DIST_INPUTS]
        
        assert action_params.shape == (batch_size, 2)
        assert action_params.dtype == torch.float32
    
    @pytest.mark.parametrize("batch_size", [1, 10, 100, 1000])
    def test_batch_size_scaling_inference(self, batch_size):
        """Test that LinearBidder scales with different batch sizes for inference."""
        observations = torch.randn(batch_size, 1, dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward_inference(batch)
        actions = result[Columns.ACTIONS]
        
        assert actions.shape == (batch_size, 1)
        assert actions.dtype == torch.float32


class TestLinearBidderGaussianDiagonal:
    """Specific tests for Gaussian Diagonal distribution format."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.bidder = LinearBidder(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        self.bidder.setup()
    
    def test_gaussian_diagonal_format(self):
        """Test that output format is correct for Gaussian Diagonal distribution."""
        observations = torch.tensor([[0.5]], dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward(batch)
        action_params = result[Columns.ACTION_DIST_INPUTS]
        
        # Should have shape (1, 2) for mean and log_std
        assert action_params.shape == (1, 2)
        
        # First column should be mean logits
        mean_logits = action_params[0, 0]
        # Second column should be log_std
        log_std = action_params[0, 1]
        
        # Check that log_std is actually log of sigma
        expected_log_std = torch.log(self.bidder.sigma)
        assert torch.isclose(log_std, expected_log_std)
    
    def test_gaussian_diagonal_parameter_ranges(self):
        """Test that Gaussian Diagonal parameters are in correct ranges."""
        # Test with different sigma values
        test_sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        for sigma_val in test_sigmas:
            with torch.no_grad():
                self.bidder.sigma.data.fill_(sigma_val)
            
            observations = torch.tensor([[0.5]], dtype=torch.float32)
            batch = {Columns.OBS: observations}
            
            result = self.bidder._forward(batch)
            action_params = result[Columns.ACTION_DIST_INPUTS]
            
            log_std = action_params[0, 1]
            expected_log_std = torch.log(torch.tensor(sigma_val))
            
            assert torch.isclose(log_std, expected_log_std)
    
    def test_gaussian_diagonal_broadcasting(self):
        """Test that log_std broadcasting works correctly."""
        batch_size = 5
        observations = torch.randn(batch_size, 1, dtype=torch.float32)
        batch = {Columns.OBS: observations}
        
        result = self.bidder._forward(batch)
        action_params = result[Columns.ACTION_DIST_INPUTS]
        
        # All rows should have the same log_std value
        log_std_values = action_params[:, 1]
        expected_log_std = torch.log(self.bidder.sigma)
        
        for log_std in log_std_values:
            assert torch.isclose(log_std, expected_log_std) 
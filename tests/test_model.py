import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from gymnasium import spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn

from auction_gym.core.model import ModelMeta, LinearModel, RandomModel


class TestModelMeta:
    """Test the ModelMeta metaclass functionality."""
    
    def test_model_registration(self):
        """Test that models are automatically registered with their model_type."""
        # Clear registry for clean test
        ModelMeta._registry.clear()
        
        # Create a test model class
        class TestModel(metaclass=ModelMeta):
            model_type = 'test_model'
        
        # Check that the model was registered
        assert 'test_model' in ModelMeta._registry
        assert ModelMeta._registry['test_model'] == TestModel
    
    
    def test_get_available_model_types(self):
        """Test getting list of available model types."""
        # Clear registry and add some test models
        ModelMeta._registry.clear()
        
        class Model1(metaclass=ModelMeta):
            model_type = 'model1'
        
        class Model2(metaclass=ModelMeta):
            model_type = 'model2'
        
        available_types = ModelMeta.get_available_model_types()
        assert 'model1' in available_types
        assert 'model2' in available_types
        assert len(available_types) == 2
    
    def test_get_model_class_success(self):
        """Test successfully getting a model class by type."""
        # Clear registry and add a test model
        ModelMeta._registry.clear()
        
        class TestModel(metaclass=ModelMeta):
            model_type = 'test_model'
        
        retrieved_class = ModelMeta.get_model_class('test_model')
        assert retrieved_class == TestModel
    
    def test_get_model_class_not_found(self):
        """Test getting a model class that doesn't exist."""
        # Clear registry
        ModelMeta._registry.clear()
        
        with pytest.raises(ValueError) as exc_info:
            ModelMeta.get_model_class('nonexistent_model')
        
        assert "Model type 'nonexistent_model' not registered" in str(exc_info.value)
        assert "Available types: []" in str(exc_info.value)


class TestLinearModel:
    """Test the LinearModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.num_outputs = 1
        self.model_config = {}
        self.name = "test_linear_model"
        
        # Create model instance
        self.model = LinearModel(
            self.obs_space, 
            self.action_space, 
            self.num_outputs, 
            self.model_config, 
            self.name
        )
    
    def test_linear_model_initialization(self):
        """Test LinearModel initialization."""
        assert self.model.model_type == 'linear'
        assert isinstance(self.model.layer, torch.nn.Linear)
        assert self.model.layer.in_features == 1
        assert self.model.layer.out_features == 1
    
    def test_linear_model_forward(self):
        """Test LinearModel forward pass."""
        # Create test input
        valuations = torch.tensor([[0.5]], dtype=torch.float32)
        input_dict = {"valuations": valuations}
        state = []
        seq_lens = None
        
        # Run forward pass
        output, new_state = self.model.forward(input_dict, state, seq_lens)
        
        # Check output shape and type
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 1)
        assert new_state == state  # State should be unchanged
    
    def test_linear_model_forward_batch(self):
        """Test LinearModel forward pass with batch input."""
        # Create batch input
        valuations = torch.tensor([[0.3], [0.7], [0.9]], dtype=torch.float32)
        input_dict = {"valuations": valuations}
        state = []
        seq_lens = None
        
        # Run forward pass
        output, new_state = self.model.forward(input_dict, state, seq_lens)
        
        # Check output shape
        assert output.shape == (3, 1)
        assert new_state == state
    
    def test_linear_model_forward_different_valuations(self):
        """Test LinearModel with different valuation inputs."""
        test_valuations = [0.0, 0.5, 1.0]
        
        for val in test_valuations:
            valuations = torch.tensor([[val]], dtype=torch.float32)
            input_dict = {"valuations": valuations}
            state = []
            seq_lens = None
            
            output, _ = self.model.forward(input_dict, state, seq_lens)
            
            # Output should be a tensor with the same shape
            assert isinstance(output, torch.Tensor)
            assert output.shape == (1, 1)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


class TestRandomModel:
    """Test the RandomModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.num_outputs = 2
        self.model_config = {}
        self.name = "test_random_model"
        
        # Create model instance
        self.model = RandomModel(
            self.obs_space, 
            self.action_space, 
            self.num_outputs, 
            self.model_config, 
            self.name
        )
    
    def test_random_model_initialization(self):
        """Test RandomModel initialization."""
        assert self.model.model_type == 'random'
        # RandomModel doesn't have any trainable parameters
        assert len(list(self.model.parameters())) == 0
    
    def test_random_model_forward(self):
        """Test RandomModel forward pass."""
        # Create test input (should be ignored by random model)
        valuations = torch.tensor([[0.5, 0.3]], dtype=torch.float32)
        input_dict = {"valuations": valuations}
        state = []
        seq_lens = None
        
        # Run forward pass
        output, new_state = self.model.forward(input_dict, state, seq_lens)
        
        # Check output shape and type
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2,)  # Should match action_space.shape
        assert new_state == state  # State should be unchanged
    
    def test_random_model_forward_batch(self):
        """Test RandomModel forward pass with batch input."""
        # Create batch input
        valuations = torch.tensor([[0.3, 0.7], [0.9, 0.1]], dtype=torch.float32)
        input_dict = {"valuations": valuations}
        state = []
        seq_lens = None
        
        # Run forward pass
        output, new_state = self.model.forward(input_dict, state, seq_lens)
        
        # Check output shape
        assert output.shape == (2,)  # Should match action_space.shape
        assert new_state == state
    
    def test_random_model_output_range(self):
        """Test that RandomModel outputs are within expected range."""
        valuations = torch.tensor([[0.5]], dtype=torch.float32)
        input_dict = {"valuations": valuations}
        state = []
        seq_lens = None
        
        # Run multiple forward passes to check randomness
        outputs = []
        for _ in range(10):
            output, _ = self.model.forward(input_dict, state, seq_lens)
            outputs.append(output)
            
            # Check that output is within [0, 1] range (torch.rand default)
            assert torch.all(output >= 0)
            assert torch.all(output <= 1)
        
        # Check that outputs are different (randomness)
        outputs_tensor = torch.stack(outputs)
        # At least some outputs should be different
        assert torch.std(outputs_tensor) > 0.01


class TestModelIntegration:
    """Integration tests for models."""
    
    def test_model_registry_integration(self):
        """Test that both models are properly registered."""
        # Clear registry for clean test
        ModelMeta._registry.clear()
        
        # Create test model classes dynamically to trigger registration
        class TestLinearModel(TorchModelV2, nn.Module, metaclass=ModelMeta):
            """Test linear model for registration testing."""
            model_type = 'linear'
            
            def __init__(self, obs_space, action_space, num_outputs, model_config, name):
                TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
                nn.Module.__init__(self)
                self.layer = nn.modules.Linear(1, 1)
                
            def forward(self, input_dict, state, seq_lens):
                x = input_dict["valuations"]
                bid = self.layer(x)
                return bid, state
        
        class TestRandomModel(TorchModelV2, nn.Module, metaclass=ModelMeta):
            """Test random model for registration testing."""
            model_type = 'random'
            
            def __init__(self, obs_space, action_space, num_outputs, model_config, name):
                TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
                nn.Module.__init__(self)
                
            def forward(self, input_dict, state, seq_lens):
                noise = torch.rand(self.action_space.shape)
                return noise, state
        
        # Now check that the models were registered
        available_types = ModelMeta.get_available_model_types()
        assert 'linear' in available_types
        assert 'random' in available_types
        
        # Test getting model classes
        linear_class = ModelMeta.get_model_class('linear')
        random_class = ModelMeta.get_model_class('random')
        
        assert linear_class == TestLinearModel
        assert random_class == TestRandomModel
    
    def test_model_forward_compatibility(self):
        """Test that both models have compatible forward interfaces."""
        obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        num_outputs = 1
        model_config = {}
        name = "test_model"
        
        # Test LinearModel
        linear_model = LinearModel(obs_space, action_space, num_outputs, model_config, name)
        linear_input = {"valuations": torch.tensor([[0.5]], dtype=torch.float32)}
        linear_output, linear_state = linear_model.forward(linear_input, [], None)
        
        # Test RandomModel
        random_model = RandomModel(obs_space, action_space, num_outputs, model_config, name)
        random_input = {"valuations": torch.tensor([[0.5]], dtype=torch.float32)}
        random_output, random_state = random_model.forward(random_input, [], None)
        
        # Both should return (output, state) tuple
        assert isinstance(linear_output, torch.Tensor)
        assert isinstance(random_output, torch.Tensor)
        assert linear_state == []
        assert random_state == []
    
    @pytest.mark.parametrize("model_class,expected_type", [
        (LinearModel, 'linear'),
        (RandomModel, 'random'),
    ])
    def test_model_types(self, model_class, expected_type):
        """Test that models have correct model_type attributes."""
        assert model_class.model_type == expected_type


class TestModelEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_linear_model_empty_input(self):
        """Test LinearModel with empty input."""
        obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        model = LinearModel(obs_space, action_space, 1, {}, "test")
        
        # Test with empty valuations (batch_size=0, features=1)
        valuations = torch.tensor([], dtype=torch.float32).reshape(0, 1)
        input_dict = {"valuations": valuations}
        
        # Empty input should produce empty output without raising an error
        output, state = model.forward(input_dict, [], None)
        
        # Check that output is also empty with correct shape (0, 1)
        assert output.shape == (0, 1)
        assert state == []
    
    def test_random_model_different_action_shapes(self):
        """Test RandomModel with different action space shapes."""
        # Test with 1D action space
        obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        action_space_1d = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        model_1d = RandomModel(obs_space, action_space_1d, 3, {}, "test_1d")
        
        valuations = torch.tensor([[0.5]], dtype=torch.float32)
        input_dict = {"valuations": valuations}
        output, _ = model_1d.forward(input_dict, [], None)
        
        assert output.shape == (3,)
        
        # Test with 2D action space
        action_space_2d = spaces.Box(low=0, high=1, shape=(2, 2), dtype=np.float32)
        model_2d = RandomModel(obs_space, action_space_2d, 4, {}, "test_2d")
        
        output, _ = model_2d.forward(input_dict, [], None)
        assert output.shape == (2, 2)

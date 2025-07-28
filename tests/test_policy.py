import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from gymnasium import spaces
from dataclasses import dataclass

from auction_gym.core.policy import BidPolicyConfig, BidPolicy
from auction_gym.core.valuation import DeterministicValuation, UniformValuation
from auction_gym.core.utility import LinearUtility, RiskAverseUtility
from auction_gym.core.model import LinearModel, RandomModel


class TestBidPolicyConfig:
    """Test the BidPolicyConfig dataclass."""
    
    def test_default_config(self):
        """Test BidPolicyConfig with default values."""
        config = BidPolicyConfig()
        
        assert config.valuation_type == "deterministic"
        assert config.utility_type == "linear"
        assert config.model_type == "linear"
        assert config.is_trainable is True
        assert config.valuation_config == {'value': 0.5}
        assert config.utility_config == {}
        assert config.model_config == {}
    
    def test_custom_config(self):
        """Test BidPolicyConfig with custom values."""
        config = BidPolicyConfig(
            valuation_type="uniform",
            utility_type="risk_averse",
            model_type="random",
            is_trainable=False,
            valuation_config={'low': 0.1, 'high': 0.9},
            utility_config={'risk_factor': 0.5},
            model_config={'hidden_size': 64}
        )
        
        assert config.valuation_type == "uniform"
        assert config.utility_type == "risk_averse"
        assert config.model_type == "random"
        assert config.is_trainable is False
        assert config.valuation_config == {'low': 0.1, 'high': 0.9}
        assert config.utility_config == {'risk_factor': 0.5}
        assert config.model_config == {'hidden_size': 64}


class TestBidPolicy:
    """Test the BidPolicy class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # Create a basic config
        self.config = BidPolicyConfig(
            valuation_type="deterministic",
            utility_type="linear",
            model_type="linear",
            is_trainable=True,
            valuation_config={'value': 0.7},
            utility_config={},
            model_config={}
        )
    
    def test_policy_initialization_with_config_object(self):
        """Test BidPolicy initialization with BidPolicyConfig object."""
        policy = BidPolicy(self.obs_space, self.action_space, self.config)
        
        assert policy.valuation_type == "deterministic"
        assert policy.utility_type == "linear"
        assert policy.model_type == "linear"
        assert policy._is_trainable is True
        assert isinstance(policy.valuation, DeterministicValuation)
        assert isinstance(policy.utility, LinearUtility)
        assert isinstance(policy.model, LinearModel)
    
    def test_policy_initialization_with_dict(self):
        """Test BidPolicy initialization with dictionary config."""
        config_dict = {
            'valuation_type': 'uniform',
            'utility_type': 'risk_averse',
            'model_type': 'random',
            'is_trainable': False,
            'valuation_config': {'low': 0.2, 'high': 0.8},
            'utility_config': {},
            'model_config': {}
        }
        
        policy = BidPolicy(self.obs_space, self.action_space, config_dict)
        
        assert policy.valuation_type == "uniform"
        assert policy.utility_type == "risk_averse"
        assert policy.model_type == "random"
        assert policy._is_trainable is False
        assert isinstance(policy.valuation, UniformValuation)
        assert isinstance(policy.utility, RiskAverseUtility)
        assert isinstance(policy.model, RandomModel)
    
    def test_create_valuation(self):
        """Test _create_valuation method."""
        policy = BidPolicy(self.obs_space, self.action_space, self.config)
        
        # Test deterministic valuation
        valuation = policy._create_valuation()
        assert isinstance(valuation, DeterministicValuation)
        assert valuation.value == 0.7
        
        # Test uniform valuation
        config_uniform = BidPolicyConfig(
            valuation_type="uniform",
            valuation_config={'low': 0.1, 'high': 0.9}
        )
        policy_uniform = BidPolicy(self.obs_space, self.action_space, config_uniform)
        valuation_uniform = policy_uniform._create_valuation()
        assert isinstance(valuation_uniform, UniformValuation)
        assert valuation_uniform.low == 0.1
        assert valuation_uniform.high == 0.9
    
    def test_create_utility(self):
        """Test _create_utility method."""
        policy = BidPolicy(self.obs_space, self.action_space, self.config)
        
        # Test linear utility
        utility = policy._create_utility()
        assert isinstance(utility, LinearUtility)
        
        # Test risk averse utility
        config_risk = BidPolicyConfig(utility_type="risk_averse")
        policy_risk = BidPolicy(self.obs_space, self.action_space, config_risk)
        utility_risk = policy_risk._create_utility()
        assert isinstance(utility_risk, RiskAverseUtility)
    
    def test_build_model(self):
        """Test _build_model method."""
        policy = BidPolicy(self.obs_space, self.action_space, self.config)
        
        # Test linear model
        model = policy._build_model()
        assert isinstance(model, LinearModel)
        assert model.model_type == 'linear'
        
        # Test random model
        config_random = BidPolicyConfig(model_type="random")
        policy_random = BidPolicy(self.obs_space, self.action_space, config_random)
        model_random = policy_random._build_model()
        assert isinstance(model_random, RandomModel)
        assert model_random.model_type == 'random'
    
    def test_get_model(self):
        """Test get_model method."""
        policy = BidPolicy(self.obs_space, self.action_space, self.config)
        model = policy.get_model()
        assert isinstance(model, LinearModel)
    
    def test_get_config(self):
        """Test get_config method."""
        policy = BidPolicy(self.obs_space, self.action_space, self.config)
        retrieved_config = policy.get_config()
        assert retrieved_config == self.config
    
    def test_is_trainable_property(self):
        """Test is_trainable property."""
        # Test trainable policy
        policy_trainable = BidPolicy(self.obs_space, self.action_space, self.config)
        assert policy_trainable.is_trainable is True
        
        # Test non-trainable policy
        config_not_trainable = BidPolicyConfig(is_trainable=False)
        policy_not_trainable = BidPolicy(self.obs_space, self.action_space, config_not_trainable)
        assert policy_not_trainable.is_trainable is False
    
    def test_get_valuation_type(self):
        """Test get_valuation_type method."""
        policy = BidPolicy(self.obs_space, self.action_space, self.config)
        assert policy.get_valuation_type() == "deterministic"
    
    def test_get_utility_type(self):
        """Test get_utility_type method."""
        policy = BidPolicy(self.obs_space, self.action_space, self.config)
        assert policy.get_utility_type() == "linear"
    
    def test_get_model_type(self):
        """Test get_model_type method."""
        policy = BidPolicy(self.obs_space, self.action_space, self.config)
        assert policy.get_model_type() == "linear"


class TestBidPolicyEdgeCases:
    """Test edge cases and error conditions for BidPolicy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    def test_invalid_valuation_type(self):
        """Test initialization with invalid valuation type."""
        config = BidPolicyConfig(valuation_type="invalid_type")
        
        with pytest.raises(ValueError) as exc_info:
            BidPolicy(self.obs_space, self.action_space, config)
        
        assert "Valuation type 'invalid_type' not registered" in str(exc_info.value)
    
    def test_invalid_utility_type(self):
        """Test initialization with invalid utility type."""
        config = BidPolicyConfig(utility_type="invalid_type")
        
        with pytest.raises(ValueError) as exc_info:
            BidPolicy(self.obs_space, self.action_space, config)
        
        assert "Utility type 'invalid_type' not registered" in str(exc_info.value)
    
    def test_invalid_model_type(self):
        """Test initialization with invalid model type."""
        config = BidPolicyConfig(model_type="invalid_type")
        
        with pytest.raises(ValueError) as exc_info:
            BidPolicy(self.obs_space, self.action_space, config)
        
        assert "Model type 'invalid_type' not registered" in str(exc_info.value)
    
    def test_invalid_valuation_config(self):
        """Test initialization with invalid valuation config."""
        config = BidPolicyConfig(
            valuation_type="deterministic",
            valuation_config={'invalid_param': 'value'}
        )
        
        with pytest.raises(TypeError):
            BidPolicy(self.obs_space, self.action_space, config)
    
    def test_empty_config_dict(self):
        """Test initialization with empty config dictionary."""
        empty_config = {}
        
        policy = BidPolicy(self.obs_space, self.action_space, empty_config)
        
        # Should use default values
        assert policy.valuation_type == "deterministic"
        assert policy.utility_type == "linear"
        assert policy.model_type == "linear"
        assert policy.is_trainable is True


class TestBidPolicyIntegration:
    """Test integration between different components of BidPolicy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    def test_valuation_utility_integration(self):
        """Test that valuation and utility work together correctly."""
        config = BidPolicyConfig(
            valuation_type="deterministic",
            utility_type="linear",
            valuation_config={'value': 0.8}
        )
        
        policy = BidPolicy(self.obs_space, self.action_space, config)
        
        # Test valuation
        valuation = policy.valuation()
        assert valuation == 0.8
        
        # Test utility calculation
        utility = policy.utility(won=True, valuation=0.8, payment=0.3)
        assert utility == 0.5  # 0.8 - 0.3 = 0.5
    
    def test_model_forward_pass(self):
        """Test that the model can perform a forward pass."""
        config = BidPolicyConfig(
            model_type="linear",
            model_config={}
        )
        
        policy = BidPolicy(self.obs_space, self.action_space, config)
        model = policy.get_model()
        
        # Create test input
        input_dict = {"valuations": torch.tensor([[0.5]], dtype=torch.float32)}
        state = []
        seq_lens = None
        
        # Perform forward pass
        output, new_state = model.forward(input_dict, state, seq_lens)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 1)  # batch_size=1, action_dim=1
        assert new_state == state  # Linear model doesn't change state
    
    def test_random_model_forward_pass(self):
        """Test that the random model can perform a forward pass."""
        config = BidPolicyConfig(
            model_type="random",
            model_config={}
        )
        
        policy = BidPolicy(self.obs_space, self.action_space, config)
        model = policy.get_model()
        
        # Create test input
        input_dict = {"valuations": torch.tensor([[0.5]], dtype=torch.float32)}
        state = []
        seq_lens = None
        
        # Perform forward pass
        output, new_state = model.forward(input_dict, state, seq_lens)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1,)  # action_space.shape
        assert new_state == state  # Random model doesn't change state


class TestBidPolicyConfigValidation:
    """Test configuration validation and edge cases."""
    
    def test_config_with_none_values(self):
        """Test config with None values."""
        config = BidPolicyConfig(
            valuation_config=None,
            utility_config=None,
            model_config=None
        )
        
        # Should not raise an error, but configs should be empty dicts
        assert config.valuation_config == {}
        assert config.utility_config == {}
        assert config.model_config == {}
    
    def test_config_with_complex_nested_dicts(self):
        """Test config with complex nested dictionaries."""
        complex_config = BidPolicyConfig(
            valuation_config={
                'nested': {
                    'deep': {
                        'value': 0.5,
                        'list': [1, 2, 3]
                    }
                }
            },
            utility_config={
                'parameters': {
                    'alpha': 0.1,
                    'beta': 0.2
                }
            },
            model_config={
                'layers': [64, 32, 16],
                'activation': 'relu'
            }
        )
        
        assert complex_config.valuation_config['nested']['deep']['value'] == 0.5
        assert complex_config.utility_config['parameters']['alpha'] == 0.1
        assert complex_config.model_config['layers'] == [64, 32, 16]


@pytest.mark.parametrize("valuation_type,utility_type,model_type", [
    ("deterministic", "linear", "linear"),
    ("uniform", "risk_averse", "random"),
    ("deterministic", "linear", "random"),
    ("uniform", "risk_averse", "linear"),
])
def test_policy_combinations(valuation_type, utility_type, model_type):
    """Test different combinations of valuation, utility, and model types."""
    obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    config = BidPolicyConfig(
        valuation_type=valuation_type,
        utility_type=utility_type,
        model_type=model_type,
        valuation_config={'value': 0.5} if valuation_type == "deterministic" else {'low': 0.0, 'high': 1.0}
    )
    
    policy = BidPolicy(obs_space, action_space, config)
    
    assert policy.valuation_type == valuation_type
    assert policy.utility_type == utility_type
    assert policy.model_type == model_type
    assert policy.is_trainable is True 
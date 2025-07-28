import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod, ABCMeta
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from ray.rllib.policy.torch_policy import TorchPolicy
from .valuation import ValuationMeta
from .utility import UtilityMeta
from .model import ModelMeta


@dataclass
class BidPolicyConfig:
    """Simple configuration dataclass for BiddingAgent."""
    valuation_type: str = "deterministic"
    utility_type: str = "linear"
    model_type: str = "linear"
    is_trainable: bool = True
    valuation_config: Dict[str, Any] = field(default_factory=lambda: {'value': 0.5})
    utility_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    

class BidPolicy(TorchPolicy):
    """Base class for bidding policies"""
    
    def __init__(self, observation_space, action_space, config):
        """
        Initialize the agent as a TorchPolicy.
        
        Args:
            observation_space: The observation space for the environment.
            action_space: The action space for the environment.
            config: Configuration dictionary or BiddingAgentConfig object containing agent parameters.
        """
        # Store spaces explicitly
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Initialize TorchPolicy
        super().__init__(observation_space, action_space, config)
        
        # Convert dict to config if needed
        if isinstance(config, dict):
            self.config = BidPolicyConfig(**config)
        else:
            self.config = config
            
        # Extract agent-specific config
        self.valuation_type = self.config.valuation_type
        self.utility_type = self.config.utility_type
        self.model_type = self.config.model_type
        self._is_trainable = self.config.is_trainable
        
        # Create valuation and utility components
        self.valuation = self._create_valuation()
        self.utility = self._create_utility()
        self.model = self._build_model()

    def _build_model(self):
        """Create the neural network model using the model registry."""
        model_class = ModelMeta.get_model_class(self.model_type)
        model = model_class(
            self.observation_space,
            self.action_space,
            self.action_space.shape[0],  # num_outputs
            self.config.model_config,
            f"{self.model_type}_auction_model"
        )
        return model
    
    def _create_valuation(self):
        """Create the valuation component."""
        valuation_class = ValuationMeta.get_valuation_class(self.valuation_type)
        return valuation_class(**self.config.valuation_config)
    
    def _create_utility(self):
        """Create the utility component."""
        utility_class = UtilityMeta.get_utility_class(self.utility_type)
        return utility_class(**self.config.utility_config)
    
    # Additional utility methods
    def get_model(self):
        """Get the neural network model."""
        return self.model
    
    def get_config(self):
        """Get the agent configuration."""
        return self.config
    
    @property
    def is_trainable(self) -> bool:
        """Check if this agent is trainable."""
        return self._is_trainable
    
    def get_valuation_type(self) -> str:
        """Get the valuation type."""
        return self.valuation_type
    
    def get_utility_type(self) -> str:
        """Get the utility type."""
        return self.utility_type
    
    def get_model_type(self) -> str:
        """Get the model type."""
        return self.model_type

    




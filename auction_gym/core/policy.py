import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod, ABCMeta
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from ray.rllib.core.rl_module.torch import TorchRLModule
from .valuation import ValuationMeta
from .utility import UtilityMeta


class LinearBidder(TorchRLModule):
    def __init__(self, observation_space, action_space, **kwargs):
        """Initialize LinearBidder with the new RLlib API."""
        # Extract required parameters for the new API
        inference_only = kwargs.pop('inference_only', False)
        model_config = kwargs.pop('model_config', {})
        catalog_class = kwargs.pop('catalog_class', None)
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs
        )
        
        # Initialize the learnable parameter theta
        # Start with theta = 1.0 (bidding the full valuation)
        
    
    def setup(self):
        """Setup method called by parent class."""
        self.theta = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    
        pass
    
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass that computes bid = observation * theta
        
        Args:
            batch: Dictionary containing the observation
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing the computed bid
        """
        # Extract observation from batch
        # The observation is typically the valuation or some representation of it
        observation = batch.get("obs", batch.get("observations"))
        
        # Handle None or missing observations
        if observation is None:
            raise ValueError("No observation provided in batch. Expected 'obs' or 'observations' key.")
        
        # Convert to tensor if it's not already
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        
        # Ensure observation has the right shape (batch_size, feature_dim)
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)  # Add batch dimension
        
        # Compute bid: observation * theta
        bid = observation * self.theta
        
        # Ensure bid is within valid range [0, 1] (clamp to action space bounds)
        bid = torch.clamp(bid, 0.0, 1.0)
        
        return {
            "action": bid,
            "action_dist_inputs": bid  # For compatibility with RLlib
        }
        

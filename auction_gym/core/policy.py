import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod, ABCMeta
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns
from .valuation import ValuationMeta
from .utility import UtilityMeta


class LinearBidder(TorchRLModule):
    def __init__(self, observation_space, action_space, **kwargs):
        """Initialize LinearBidder with the new RLlib API."""
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
        

    
    def setup(self):
        """Setup method called by parent class."""
        self.mu = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.sigma = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))


    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass that computes mean and log_std for Gaussian Diagonal distribution
        
        Args:
            batch: Dictionary containing the observation
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing the computed action distribution parameters
        """
        observations = batch[Columns.OBS]  # Shape should be (B,1)
        
        # Compute mean (logits) - this will be the center of the distribution
        mean_logits = self.mu * observations  # Shape: (B,1)
        
        # Compute log_std - this controls the spread of the distribution
        # For Gaussian Diagonal, we need log_std, not std
        log_std = torch.log(self.sigma)  # Convert std to log_std
        log_std = log_std.expand_as(mean_logits)  # Broadcast to match batch size
        
        # Concatenate mean and log_std along the last dimension
        action_params = torch.cat([mean_logits, log_std], dim=1)  # Shape: (B,2)
        
        return {Columns.ACTION_DIST_INPUTS: action_params}

    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        observations = batch[Columns.OBS]  # Shape should be (B,1)
        return {Columns.ACTIONS: self.mu *observations}

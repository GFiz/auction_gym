from ray.rllib.models.torch.torch_distributions import TorchDistribution
import torch
from typing import Dict, Any, Optional, Type
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns
from auction_gym.core.distributions import TorchUnitIntervalGaussian


class LBTorchRLModule(TorchRLModule):
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
        self.initial_mu = self.model_config.get('initial_mu', 1.0) or 1.0
        self.initial_sigma = self.model_config.get('initial_sigma', 0.0) or 0.0
        self.mu = torch.nn.Parameter(torch.tensor(self.initial_mu, dtype=torch.float32))
        self.log_sigma = torch.nn.Parameter(torch.tensor(self.initial_sigma, dtype=torch.float32))


    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass that computes mean and log_std for Gaussian Diagonal distribution
        
        Args:
            batch: Dictionary containing the observation
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing the computed action distribution parameters
        """
        observations = batch[Columns.OBS]  # Shape should be (B,1) or (B,)
        
        # Ensure observations have the right shape
        if observations.dim() == 1:
            observations = observations.unsqueeze(-1)  # Make it (B,1)
        
        # Compute mean (logits) - this will be the center of the distribution
        mean_logits = self.mu * observations  # Shape: (B,1)
        
        # Get log_std from the log_sigma parameter
        log_sigma = self.log_sigma.expand_as(mean_logits)  # Broadcast to match batch size
        
        # Concatenate mean and log_std along the last dimension
        action_params = torch.cat([mean_logits, log_sigma], dim=-1)  # Shape: (B,2)
        
        return {Columns.ACTION_DIST_INPUTS: action_params}

    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        observations = batch[Columns.OBS]  # Shape should be (B,1) or (B,)
        
        # Ensure observations have the right shape
        if observations.dim() == 1:
            observations = observations.unsqueeze(-1)  # Make it (B,1)
            
        return {Columns.ACTIONS: self.mu * observations}
    
    def get_inference_action_dist_cls(self) -> Type[TorchDistribution]:
        return TorchUnitIntervalGaussian
    

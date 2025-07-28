# Remove incorrect import
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import torch
from abc import ABCMeta
from gymnasium import spaces
from typing import Dict, Any, Optional

class ModelMeta(ABCMeta):
    """Metaclass for automatic agent type registration."""
    
    _registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        if bases and not getattr(cls, '__abstractmethods__', None):
            if 'model_type' in namespace:
                model_type = namespace['model_type']
                mcs._registry[model_type] = cls
        
        return cls
    
    @classmethod
    def get_available_model_types(mcs) -> list[str]:
        return list(mcs._registry.keys())
    
    @classmethod
    def get_model_class(mcs, model_type: str) -> type:
        if model_type not in mcs._registry:
            raise ValueError(f"Model type '{model_type}' not registered. Available types: {list(mcs._registry.keys())}")
        return mcs._registry[model_type]

class LinearAuctionModel(TorchModelV2, nn.Module):
    """Simple linear model for linear bidding agents."""
    model_type = 'linear'
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.layer = nn.modules.Linear(1,1)
        
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["valuations"]
        bid = self.layer(x)
        return bid, state
    
    def value_function(self):
        return self.value_head(self._cur_features)


class RandomAuctionModel(TorchModelV2, nn.Module):
    """Model that outputs random actions."""
    model_type = 'random'
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
    def forward(self, input_dict, state, seq_lens):
        noise = torch.randn_like(self.action_space.shape) 
        return noise, state
    
    def value_function(self):
        return self.value_head(self._cur_features)


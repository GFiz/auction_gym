from abc import ABC, abstractmethod, ABCMeta
import numpy as np
import torch
from torch.distributions import Distribution, Uniform, Normal, Gamma
from typing import Optional


class ValuationMeta(ABCMeta):
    """Metaclass for automatic valuation type registration."""
    
    _registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        if bases and not getattr(cls, '__abstractmethods__', None):
            if 'valuation_type' in namespace:
                valuation_type = namespace['valuation_type']
                mcs._registry[valuation_type] = cls
        
        return cls
    
    @classmethod
    def get_available_valuation_types(mcs) -> list[str]:
        return list(mcs._registry.keys())
    
    @classmethod
    def get_valuation_class(mcs, valuation_type: str) -> type:
        if valuation_type not in mcs._registry:
            raise ValueError(f"Valuation type '{valuation_type}' not registered. Available types: {list(mcs._registry.keys())}")
        return mcs._registry[valuation_type]


class BaseValuation(ABC, metaclass=ValuationMeta):
    """Abstract base class for valuation models."""
    
    @abstractmethod 
    def get_distribution(self, observation: Optional[np.ndarray] = None) -> Distribution:
        pass
    
    def __call__(self, observation: Optional[np.ndarray] = None) -> float:
        dist = self.get_distribution(observation)
        sample = dist.sample()
        return max(0.0, float(sample.item()))
    
class DeterministicValuation(BaseValuation):
    valuation_type = "deterministic"
    
    def __init__(self, value: float):
        if value < 0:
            raise ValueError("Valuation must be a positive real number.")
        self.value = value
    
    def get_distribution(self, observation: Optional[np.ndarray] = None) -> Distribution:
        raise NotImplementedError("DeterministicValuation does not use a distribution.")
    
    def __call__(self, observation: Optional[np.ndarray] = None) -> float:
        return self.value


class UniformValuation(BaseValuation):
    valuation_type = "uniform"
    
    def __init__(self, low: float = 0.0, high: float = 1.0):
        if low < 0 or high < 0:
            raise ValueError("Valuation bounds must be positive.")
        if high < low:
            raise ValueError("High bound must be greater than or equal to low bound.")
        self.low = torch.tensor(low, dtype=torch.float32)
        self.high = torch.tensor(high, dtype=torch.float32)

    def get_distribution(self, observation: Optional[np.ndarray] = None) -> Distribution:
        return Uniform(self.low, self.high)


class NormalValuation(BaseValuation):
    valuation_type = "normal"
    
    def __init__(self, mean: float, std: float):
        if std < 0:
            raise ValueError("Standard deviation cannot be negative.")
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
    
    def get_distribution(self, observation: Optional[np.ndarray] = None) -> Distribution:
        return Normal(self.mean, self.std)

class GammaValuation(BaseValuation):
    valuation_type = "gamma"
    
    def __init__(self, shape: float, scale: float = 1.0):
        if shape <= 0:
            raise ValueError("Shape parameter must be positive.")
        if scale <= 0:
            raise ValueError("Scale parameter must be positive.")
        self.shape = torch.tensor(shape, dtype=torch.float32)
        self.scale = torch.tensor(scale, dtype=torch.float32)
    
    def get_distribution(self, observation: Optional[np.ndarray] = None) -> Distribution:
        # PyTorch Gamma uses concentration (shape) and rate (1/scale)
        rate = 1.0 / self.scale
        return Gamma(concentration=self.shape, rate=rate)
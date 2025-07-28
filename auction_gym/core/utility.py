from abc import ABC, abstractmethod, ABCMeta


class UtilityMeta(ABCMeta):
    """Metaclass for automatic utility type registration."""
    
    _registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        if bases and not getattr(cls, '__abstractmethods__', None):
            if 'utility_type' in namespace:
                utility_type = namespace['utility_type']
                mcs._registry[utility_type] = cls
        
        return cls
    
    @classmethod
    def get_available_utility_types(mcs) -> list[str]:
        return list(mcs._registry.keys())
    
    @classmethod
    def get_utility_class(mcs, utility_type: str) -> type:
        if utility_type not in mcs._registry:
            raise ValueError(f"Utility type '{utility_type}' not registered. Available types: {list(mcs._registry.keys())}")
        return mcs._registry[utility_type]


class BaseUtility(ABC, metaclass=UtilityMeta):
    """Abstract base class for agent utility functions."""  
    @abstractmethod
    def __call__(self, won: bool, valuation: float, payment: float) -> float:
        pass
      

class LinearUtility(BaseUtility):
    utility_type = "linear"
    
    def __call__(self, won: bool, valuation: float, payment: float) -> float:
        if won:
            return valuation - payment
        else:
            return 0.0


class RiskAverseUtility(BaseUtility):
    utility_type = "risk_averse"
    
    def __call__(self, won: bool, valuation: float, payment: float) -> float:
        if won:
            import numpy as np
            return np.log(1 + valuation - payment)
        else:
            return 0.0
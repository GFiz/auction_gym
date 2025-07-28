import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod, ABCMeta
from typing import Union, Callable, Dict, Any
from .valuation import BaseValuation, DeterministicValuation
from .utility import BaseUtility, LinearUtility


class BidderMeta(ABCMeta):
    """Metaclass for automatic agent type registration."""
    
    # Registry is stored in the metaclass, not the base class
    _registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Only register concrete classes (not abstract base classes)
        if bases and not getattr(cls, '__abstractmethods__', None):
            # Try to get agent_type from class attributes first
            if 'agent_type' in namespace:
                agent_type = namespace['agent_type']
                mcs._registry[agent_type] = cls
        
        return cls
    
    @classmethod
    def get_available_agent_types(mcs) -> list[str]:
        """Get all registered agent types."""
        return list(mcs._registry.keys())
    
    @classmethod
    def get_agent_class(mcs, agent_type: str) -> type:
        """Get agent class for a given type."""
        if agent_type not in mcs._registry:
            raise ValueError(f"Agent type '{agent_type}' not registered. Available types: {list(mcs._registry.keys())}")
        return mcs._registry[agent_type]
    

class BaseBidder(ABC, metaclass=BidderMeta):
    """Abstract base class for auction agents."""
    
    def __init__(self, action_space: spaces.Space, valuation: BaseValuation, utility: BaseUtility = LinearUtility(), is_trainable: bool = True):
        """
        Initialize the agent.
        
        Args:
            action_space: The action space for this agent.
            valuation: The valuation model for the agent.
            utility: The utility function for the agent (defaults to LinearUtility).
            is_trainable: Whether this agent can be trained by RLlib (defaults to True).
        """
        self.action_space = action_space
        self.valuation = valuation
        self.utility = utility
        self.is_trainable = is_trainable
        
    @abstractmethod
    def act(self, observation: np.ndarray) -> float:
        """
        Choose an action based on the current observation.
        
        Args:
            observation: Current environment observation.
            
        Returns:
            The chosen action (bid amount).
        """
        raise NotImplementedError
    
    def get_valuation(self, observation: np.ndarray) -> float:
        """
        Get the agent's valuation for the current item.
        
        Args:
            observation: The current environment observation.
            
        Returns:
            The agent's valuation.
        """
        return self.valuation(observation)
    
    def calculate_utility(self, won: bool, valuation: float, payment: float) -> float:
        """
        Calculate the agent's utility given the auction outcome.
        
        Args:
            won: Whether the agent won the auction.
            valuation: The agent's valuation for the item.
            payment: The amount the agent has to pay.
            
        Returns:
            The agent's utility (reward).
        """
        return self.utility(won=won, valuation=valuation, payment=payment)
    
    def reset(self) -> None:
        """Reset the agent's internal state."""
        pass
    


class RandomBidder(BaseBidder):
    """Agent that chooses actions randomly from the action space."""
    agent_type = "random"
    def __init__(self, action_space: spaces.Space, valuation: BaseValuation, utility: BaseUtility = LinearUtility(), is_trainable: bool = True):
        super().__init__(action_space, valuation, utility, is_trainable)

    def act(self, observation: np.ndarray) -> float:
        """
        Choose a random action from the action space.
        
        Args:
            observation: Current environment observation (unused).
            
        Returns:
            Random action from the action space.
        """
        return self.action_space.sample() 


class LinearBidder(BaseBidder):
    """Agent that bids linearly: bid = lambda * valuation."""
    agent_type = "linear"
    def __init__(self, action_space: spaces.Space, valuation: BaseValuation, lambda_param: float = 1.0, utility: BaseUtility = LinearUtility(), is_trainable: bool = True):
        """
        Initialize linear agent.
        
        Args:
            action_space: The action space for this agent.
            valuation: The valuation model for the agent.
            lambda_param: Linear parameter for bidding (bid = lambda * valuation).
            utility: The utility function for the agent (defaults to LinearUtility).
            is_trainable: Whether this agent can be trained by RLlib (defaults to True).
        """
        super().__init__(action_space, valuation, utility, is_trainable)
        self.lambda_param = lambda_param
    
    def act(self, observation: np.ndarray) -> float:
        """
        Bid linearly: bid = lambda * valuation.
        
        Args:
            observation: Current environment observation.
            
        Returns:
            Linear bid amount.
        """
        valuation = self.get_valuation(observation)
        bid = self.lambda_param * valuation
        
        # Ensure bid is within action space bounds
        if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
            bid = np.clip(bid, self.action_space.low, self.action_space.high)
        
        # Convert to float properly to avoid numpy deprecation warning
        if hasattr(bid, 'item'):
            return bid.item()
        return float(bid)



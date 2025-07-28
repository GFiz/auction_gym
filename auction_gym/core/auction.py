# auction_gym/core/auction.py
from abc import ABC, abstractmethod, ABCMeta
from typing import List, Tuple, Optional
import numpy as np


class AuctionMechanismMeta(ABCMeta):
    """Metaclass for automatic auction mechanism registration."""
    
    _registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Only register concrete classes
        if bases and not getattr(cls, '__abstractmethods__', None):
            if 'mechanism_type' in namespace:
                mechanism_type = namespace['mechanism_type']
                mcs._registry[mechanism_type] = cls
        
        return cls
    
    @classmethod
    def get_available_mechanism_types(mcs) -> list[str]:
        return list(mcs._registry.keys())
    
    @classmethod
    def get_mechanism_class(mcs, mechanism_type: str) -> type:
        if mechanism_type not in mcs._registry:
            raise ValueError(f"Mechanism type '{mechanism_type}' not registered. Available types: {list(mcs._registry.keys())}")
        return mcs._registry[mechanism_type]


class AuctionMechanism(ABC, metaclass=AuctionMechanismMeta):
    """Abstract base class for auction mechanisms."""
    
    @abstractmethod
    def allocate(self, bids: List[float]) -> Tuple[int, float]:
        pass
    
    @abstractmethod
    def payment(self, bids: List[float], winner: int) -> float:
        pass
    
    def run_auction(self, bids: List[float]) -> Tuple[int, float]:
        winner, allocation = self.allocate(bids)
        payment = self.payment(bids, winner)
        return winner, payment
    
    # Class methods that delegate to the metaclass
    @classmethod
    def get_available_mechanism_types(cls) -> list[str]:
        return cls.__class__.get_available_mechanism_types()
    
    @classmethod
    def get_mechanism_class(cls, mechanism_type: str) -> type:
        return cls.__class__.get_mechanism_class(mechanism_type)


class SecondPriceAuction(AuctionMechanism):
    mechanism_type = "second_price"
    
    def allocate(self, bids: List[float]) -> Tuple[int, float]:
        if not bids:
            return None, 0.0
        winner = np.argmax(bids)
        return winner, 1.0
    
    def payment(self, bids: List[float], winner: int) -> float:
        if not bids or winner is None:
            return 0.0
        if len(bids) == 1:
            return 0.0
        other_bids = [bid for i, bid in enumerate(bids) if i != winner]
        return max(other_bids)


class FirstPriceAuction(AuctionMechanism):
    mechanism_type = "first_price"
    
    def allocate(self, bids: List[float]) -> Tuple[int, float]:
        if not bids:
            return None, 0.0
        winner = np.argmax(bids)
        return winner, 1.0
    
    def payment(self, bids: List[float], winner: int) -> float:
        if not bids or winner is None:
            return 0.0
        return bids[winner]
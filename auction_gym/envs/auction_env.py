import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from auction_gym.core.mechanism import AuctionMechanism, SecondPriceAuction
from auction_gym.core.valuation import BaseValuation
from auction_gym.core.utility import BaseUtility

class AuctionEnv(MultiAgentEnv):
    """Real-Time Bidding auction environment using specified mechanisms.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        valuations: Dict[str, BaseValuation],
        utilities: Dict[str, BaseUtility],
        action_spaces: Dict[str, spaces],
        observation_spaces: Dict[str, spaces],
        mechanism: AuctionMechanism = SecondPriceAuction(),
        max_steps: int = 100,
    ):
        """
        Initialize the auction environment.
        
        Args:
            valuations: Dictionary mapping agent keys to BaseValuation instances
            utilities: Dictionary mapping agent keys to BaseUtility instances
            action_spaces: Dictionary mapping agent keys to action spaces
            observation_spaces: Dictionary mapping agent keys to observation spaces
            mechanism: Auction mechanism to use
            max_steps: Maximum number of steps per episode
        """
        super().__init__()
        
        # Check that all four dictionaries have the same keys
        valuation_keys = set(valuations.keys())
        utility_keys = set(utilities.keys())
        action_keys = set(action_spaces.keys())
        observation_keys = set(observation_spaces.keys())
        if valuation_keys!=utility_keys  or utility_keys!= action_keys or action_keys != observation_keys:
            raise ValueError(
                f"All dictionaries must have the same agent keys. "
                f"Found: valuations={valuation_keys}, utilities={utility_keys}, "
                f"action_spaces={action_keys}, observation_spaces={observation_keys}"
            )

        self.valuations = valuations
        self.utilities = utilities
        self.action_spaces = action_spaces
        self.observation_spaces = observation_spaces
                
        self.n_agents = len(self.action_spaces)
        self.mechanism = mechanism
        self.max_steps = max_steps
        
        # Multi-agent interface attributes
        self.agents = list(action_spaces.keys()) 
        
        # Environment state
        self.current_step = 0
        self.episode_data = {
            "bids": [],
            "valuations": [],
            "allocations": [],
            "payments": []
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_data = {
            "bids": [],
            "valuations": [],
            "allocations": [],
            "payments": []
        }
        
        # Get valuations for this round 
        obs_for_valuation = self._get_observation_valuation()
        valuations = np.array([valuation(obs_for_valuation) for valuation in self.valuations.values()])
        
        # Generate initial observation

        obs = self._get_observation(valuations)
        info = self._get_info()
        
        
        return obs, info

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Execute one time step within the environment."""
        # Validate actions and convert to bids
        bids = np.array([action_dict[agent_key] for agent_key in self.agents])  # Convert to numpy array
        # Run auction
        allocations, payments = self.mechanism.run_auction(bids)
        
        # Get valuations for this round 
        obs_for_valuation = self._get_observation_valuation()
        valuations = np.array([valuation(obs_for_valuation) for valuation in self.valuations.values()])
        
        # Compute rewards using policy utility functions
        rewards = self._compute_rewards(allocations, payments, valuations)
        
        # Update environment state
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Store episode data
        self.episode_data["bids"].append(bids)
        self.episode_data["valuations"].append(valuations.tolist())
        self.episode_data["allocations"].append(allocations)
        self.episode_data["payments"].append(payments)
        
        # Generate new observation and info
        obs = self._get_observation(valuations)
        info = self._get_info(allocations, payments, bids, valuations)
        
        
        terminated_dict = {agent_key: terminated for agent_key in self.agents}
        terminated_dict["__all__"] = terminated
        truncated_dict = {agent_key: truncated for agent_key in self.agents}
        truncated_dict["__all__"] = truncated
        info = info
        
        return obs, rewards, terminated_dict, truncated_dict, info

    

    def _compute_rewards(self, allocations: np.ndarray, payments: np.ndarray, valuations: np.ndarray) -> np.ndarray:
        """Calculate rewards for each agent using their own utility functions."""
        rewards = {}
        # Compute rewards vectorized
        for i, agent_key in enumerate(self.agents):
            rewards[agent_key] = self.utilities[agent_key](
                won=bool(allocations[i]),
                valuation=valuations[i],
                payment=payments[i]
            )
        
        return rewards
  
    def _get_observation_valuation(self) -> np.ndarray: #TODO This here should essentially return whatever features are used in valuation estimation
        return None
    
    def _get_observation(self,valuations) -> Dict[str,np.float32]:
        """Generate the current observation for the agents."""
        # This is a placeholder. In a real environment, this would be meaningful data.
        # Could include: time step, market conditions, historical data, etc.
        return {agent_key: valuations[i] for i,agent_key in enumerate(self.agents)}

    def _get_info(self, allocations: Optional[np.ndarray] = None, payments: Optional[np.ndarray] = None, 
                  bids: Optional[np.ndarray] = None, valuations: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Generate info dictionary for the current step."""
        info = {
            "step": self.current_step,
            "max_steps": self.max_steps,
            "n_agents": self.n_agents,
            "mechanism_type": self.mechanism.mechanism_type
        }
        
        # Add step-specific info if provided
        if allocations is not None:
            info["allocations"] = allocations
            info["payments"] = payments
        if bids is not None:
            info["bids"] = bids
        if valuations is not None:
            info["valuations"] = valuations
            
        return info

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get a summary of the current episode."""
        if not self.episode_data["bids"]:
            return {"error": "No episode data available"}
        
        # Calculate summary statistics
        total_bids = len(self.episode_data["bids"])
        total_revenue = sum(sum(payments) for payments in self.episode_data["payments"])
        avg_bid = np.mean([np.mean(bids) for bids in self.episode_data["bids"]])
        
        # Winner distribution
        winner_dist = self._get_winner_distribution()
        
        # Agent performance using actual agent keys
        agent_performance = {}
        for i,agent_key in enumerate(self.agents):
            agent_wins = sum(1 for allocations in self.episode_data["allocations"] if allocations[i])
            agent_payments = sum(payments[i] for payments in self.episode_data["payments"])
            agent_valuations = sum(valuations[i] for valuations in self.episode_data["valuations"])
            
            agent_performance[agent_key] = {
                "wins": agent_wins,
                "total_payments": agent_payments,
                "total_valuations": agent_valuations,
                "win_rate": agent_wins / total_bids if total_bids > 0 else 0.0
            }
        
        return {
            "episode_length": total_bids,
            "total_revenue": total_revenue,
            "average_bid": avg_bid,
            "winner_distribution": winner_dist,
            "agent_performance": agent_performance,
            "mechanism_type": self.mechanism.mechanism_type
        }

    def _get_winner_distribution(self) -> Dict[str, int]:
        """Calculate the distribution of winners across the episode."""
        winner_counts = {agent_key: 0 for agent_key in self.agents}
        
        for allocations in self.episode_data["allocations"]:
            for i,agent_key in enumerate(self.agents):
                if allocations[i]:
                    winner_counts[agent_key] += 1
                    break  # Only one winner per round
        
        return winner_counts

    def render(self, mode: str = "human"):
        """Render the current state of the environment."""
        if mode != "human":
            raise ValueError(f"Unsupported render mode: {mode}")
        
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Mechanism: {self.mechanism.mechanism_type}")
        print(f"Number of agents: {self.n_agents}")
        
        if self.episode_data["bids"]:
            print(f"Last round bids: {self.episode_data['bids'][-1]}")
            print(f"Last round allocations: {self.episode_data['allocations'][-1]}")
            print(f"Last round payments: {self.episode_data['payments'][-1]}")
            print(f"Last round valuations: {self.episode_data['valuations'][-1]}")
        
        print(f"Winner distribution: {self._get_winner_distribution()}")

    def close(self):
        """Clean up any resources."""
        pass 
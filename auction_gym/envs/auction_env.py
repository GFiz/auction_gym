import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from auction_gym.core.mechanism import AuctionMechanism, SecondPriceAuction
from auction_gym.core.policy import BidPolicy


class AuctionEnv(MultiAgentEnv):
    """Real-Time Bidding auction environment using auction mechanisms.
    
    Supports multi-agent RLlib interface directly.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        policies: List[BidPolicy],
        mechanism: AuctionMechanism = SecondPriceAuction(),
        max_steps: int = 100,
    ):
        """
        Initialize the auction environment.
        
        Args:
            policies: List of BidPolicy instances (one per agent)
            mechanism: Auction mechanism to use
            max_steps: Maximum number of steps per episode
            observation_space: Custom observation space (optional)
        """
        super().__init__()
        
        if not policies:
            raise ValueError("At least one policy must be provided.")
        
        self.policies = policies  # Keep original policy objects
        self.n_agents = len(policies)
        self.mechanism = mechanism
        self.max_steps = max_steps
        

        # Multi-agent spaces (required by RLlib API) - retrieve from policies
        self.action_spaces = {f"agent_{i}": policy.action_space for i, policy in enumerate(self.policies)}
        self.observation_spaces = {f"agent_{i}": policy.observation_space for i, policy in enumerate(self.policies)}
        
        # Multi-agent interface attributes
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agents = self.possible_agents.copy()  # All agents always active
        
        # Environment state
        self.current_step = 0
        self.episode_data = {
            "bids": [],
            "valuations": [],
            "winners": [],
            "payments": []
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_data = {
            "bids": [],
            "valuations": [],
            "winners": [],
            "payments": []
        }
        
        # Generate initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        # Return multi-agent format
        obs_dict = {f"agent_{i}": obs for i in range(self.n_agents)}
        info_dict = {f"agent_{i}": info for i in range(self.n_agents)}
        
        return obs_dict, info_dict

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Execute one time step within the environment."""
        # Validate actions and convert to bids
        bids = self._process_actions(action_dict)
        
        # Run auction
        winner_idx, payment = self.mechanism.run_auction(bids)
        
        # Get valuations for this round using policy valuation functions
        obs_for_valuation = self._get_observation()
        valuations = np.array([policy.valuation(obs_for_valuation) for policy in self.policies])
        
        # Compute rewards using policy utility functions
        rewards = self._compute_rewards(winner_idx, payment, valuations)
        
        # Update environment state
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Store episode data
        self.episode_data["bids"].append(bids)
        self.episode_data["valuations"].append(valuations.tolist())
        self.episode_data["winners"].append(winner_idx)
        self.episode_data["payments"].append(payment)
        
        # Generate new observation and info
        obs = self._get_observation()
        info = self._get_info(winner=winner_idx, payment=payment, bids=bids, valuations=valuations)
        
        # Return multi-agent format
        obs_dict = {f"agent_{i}": obs for i in range(self.n_agents)}
        rewards_dict = {f"agent_{i}": rewards[i] for i in range(self.n_agents)}
        terminated_dict = {f"agent_{i}": terminated for i in range(self.n_agents)}
        terminated_dict["__all__"] = terminated
        truncated_dict = {f"agent_{i}": truncated for i in range(self.n_agents)}
        truncated_dict["__all__"] = truncated
        info_dict = {f"agent_{i}": info for i in range(self.n_agents)}
        
        return obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict

    def _process_actions(self, action_dict: Dict[str, Any]) -> np.ndarray:
        """Process actions from policies and convert to bids vector."""
        bids = np.zeros(self.n_agents, dtype=np.float32)
        
        for i in range(self.n_agents):
            agent_id = f"agent_{i}"
            if agent_id not in action_dict:
                raise ValueError(f"Missing action for {agent_id}")
            
            action = action_dict[agent_id]
            
            # Convert action to float bid
            if isinstance(action, np.ndarray):
                if action.shape == (1,):
                    bids[i] = float(action[0])
                elif action.shape == ():  # scalar
                    bids[i] = float(action)
                else:
                    raise ValueError(f"Expected action shape (1,) or (), got {action.shape}")
            else:
                bids[i] = float(action)
        
        return bids

    def _compute_rewards(self, allocations: Optional[int], payments: float, valuations: np.ndarray) -> np.ndarray:
        """Calculate rewards for each agent using their own utility functions."""
        rewards = np.zeros(self.n_agents, dtype=np.float32)

        # Compute rewards vectorized
        for i in range(self.n_agents):
            rewards[i] = self.policies[i].utility(
                won=allocations[i],
                valuation=valuations[i],
                payment=payments[i]
            )
        
        return rewards

    def _get_observation(self) -> np.ndarray:
        """Generate the current observation for the agents."""
        # This is a placeholder. In a real environment, this would be meaningful data.
        # Could include: time step, market conditions, historical data, etc.
        return np.array([self.current_step / self.max_steps], dtype=np.float32)

    def _get_info(self, **kwargs) -> Dict[str, Any]:
        """Get info dictionary for the current step."""
        info = {
            "step": self.current_step,
            "max_steps": self.max_steps,
            "mechanism_type": type(self.mechanism).__name__
        }
        
        # Add auction-specific info if provided
        if "winner" in kwargs:
            info["winner"] = kwargs["winner"]
        if "payment" in kwargs:
            info["payment"] = kwargs["payment"]
        if "bids" in kwargs:
            info["bids"] = kwargs["bids"]
        if "valuations" in kwargs:
            info["valuations"] = kwargs["valuations"]
        
        return info

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the current episode."""
        if not self.episode_data["bids"]:
            return {}
        
        bids_array = np.array(self.episode_data["bids"])
        valuations_array = np.array(self.episode_data["valuations"])
        
        summary = {
            "total_steps": len(self.episode_data["bids"]),
            "total_revenue": sum(self.episode_data["payments"]),
            "avg_bid": np.mean(bids_array),
            "avg_valuation": np.mean(valuations_array),
            "winning_bids": [self.episode_data["bids"][i][self.episode_data["winners"][i]] 
                           for i in range(len(self.episode_data["winners"])) 
                           if self.episode_data["winners"][i] is not None],
            "winner_distribution": self._get_winner_distribution()
        }
        
        return summary

    def _get_winner_distribution(self) -> Dict[str, int]:
        """Get distribution of winners across agents."""
        distribution = {f"agent_{i}": 0 for i in range(self.n_agents)}
        for winner in self.episode_data["winners"]:
            if winner is not None:
                distribution[f"agent_{winner}"] += 1
        return distribution

    def render(self, mode: str = "human"):
        """Render the environment state."""
        if mode == "human":
            print(f"--- Step: {self.current_step}/{self.max_steps} ---")
            print(f"  Auction: {type(self.mechanism).__name__}")
            print(f"  Agents: {self.n_agents}")
            
            if self.episode_data["bids"]:
                latest_bids = self.episode_data["bids"][-1]
                latest_valuations = self.episode_data["valuations"][-1]
                latest_winner = self.episode_data["winners"][-1]
                latest_payment = self.episode_data["payments"][-1]
                
                print(f"  Latest bids: {latest_bids}")
                print(f"  Latest valuations: {latest_valuations}")
                print(f"  Winner: agent_{latest_winner}" if latest_winner is not None else "  Winner: None")
                print(f"  Payment: {latest_payment}")

    def close(self):
        """Clean up any resources."""
        pass 
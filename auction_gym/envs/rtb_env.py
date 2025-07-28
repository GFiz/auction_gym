import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from auction_gym.core.auction import AuctionMechanism, SecondPriceAuction
from auction_gym.core.agent import BaseAgent


class RTBAuctionEnv(MultiAgentEnv):
    """Real-Time Bidding auction environment using auction mechanisms.
    
    Supports multi-agent RLlib interface directly.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        agents: List[BaseAgent],
        auction_mechanism: AuctionMechanism = SecondPriceAuction() 
    ):
        super().__init__()
        
        if not agents:
            raise ValueError("At least one agent must be provided.")
        self.agent_objects = agents  # Keep original agent objects
        self.n_agents = len(agents)
        
        # Observation space: For now, a placeholder. Could be anything.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Multi-agent spaces (required by new RLLib API) - use agents' existing action spaces
        self.action_spaces = {f"agent_{i}": agent.action_space for i, agent in enumerate(self.agent_objects)}
        self.observation_spaces = {f"agent_{i}": self.observation_space for i in range(self.n_agents)}
        
        # Multi-agent interface attributes - all agents that could potentially appear
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        # For auction env, all agents are always active throughout episode (agent IDs)
        self.agents = self.possible_agents.copy()
        
        # Auction mechanism
        self.auction_mechanism = auction_mechanism
        
        # Environment state
        self.current_step = 0
        self.max_steps = 100

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        
        # In a real scenario, you might generate a new observation here
        obs = self._get_observation()
        info = self._get_info()
        
        # Return multi-agent format: dictionaries mapping agent IDs to their values
        obs_dict = {f"agent_{i}": obs for i in range(self.n_agents)}
        info_dict = {f"agent_{i}": info for i in range(self.n_agents)}
        
        return obs_dict, info_dict

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Execute one time step within the environment."""
        # Convert action dict to list of bids
        bids = []
        for i in range(self.n_agents):
            agent_id = f"agent_{i}"
            if agent_id in action_dict:
                action = action_dict[agent_id]
                # Handle different action formats
                if isinstance(action, np.ndarray):
                    if action.shape == (1,):
                        bids.append(float(action[0]))
                    elif action.shape == ():  # scalar
                        bids.append(float(action))
                    else:
                        raise ValueError(f"Expected action shape (1,) or (), got {action.shape}")
                else:
                    bids.append(float(action))
            else:
                raise ValueError(f"Missing action for {agent_id}")
        
        winner_idx, payment = self.auction_mechanism.run_auction(bids)
        
        # Get valuations for this round
        obs_for_valuation = self._get_observation()
        valuations = np.array([agent.get_valuation(obs_for_valuation) for agent in self.agent_objects])
        
        rewards = self._calculate_rewards(winner_idx, payment, valuations)
        
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        obs = self._get_observation()
        info = self._get_info(winner=winner_idx, payment=payment, bids=bids, valuations=valuations)
        
        # Return multi-agent format: all as dictionaries
        obs_dict = {f"agent_{i}": obs for i in range(self.n_agents)}
        rewards_dict = {f"agent_{i}": rewards[i] for i in range(self.n_agents)}
        terminated_dict = {f"agent_{i}": terminated for i in range(self.n_agents)}
        terminated_dict["__all__"] = terminated  # Special key for environment termination
        truncated_dict = {f"agent_{i}": truncated for i in range(self.n_agents)}
        truncated_dict["__all__"] = truncated  # Special key for environment truncation
        info_dict = {f"agent_{i}": info for i in range(self.n_agents)}
        
        return obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict

    def _calculate_rewards(self, winner_idx: Optional[int], payment: float, valuations: np.ndarray) -> np.ndarray:
        """Calculate rewards for each agent using their own utility functions."""
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            won = (i == winner_idx)
            agent_payment = payment if won else 0.0
            rewards[i] = self.agent_objects[i].calculate_utility(
                won=won,
                valuation=valuations[i],
                payment=agent_payment
            )
        return rewards

    def _get_observation(self) -> np.ndarray:
        """Generate the current observation for the agents."""
        # This is a placeholder. In a real environment, this would be meaningful data.
        return np.array([self.current_step / self.max_steps], dtype=np.float32)

    def _get_info(self, **kwargs) -> Dict[str, Any]:
        """Get info dictionary for the current step."""
        info = {
            "step": self.current_step,
            "auction_mechanism": type(self.auction_mechanism).__name__
        }
        info.update(kwargs)
        return info

    def render(self, mode: str = "human"):
        """Render the environment state."""
        if mode == "human":
            print(f"--- Step: {self.current_step} ---")
            print(f"  Auction: {type(self.auction_mechanism).__name__}")

    def close(self):
        """Clean up any resources."""
        pass 
"""
Experiment class for training agents using RLlib and auction gym.
"""

import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
from gymnasium import spaces
from typing import Dict, Any, List, Optional, Callable
import os
import json
from datetime import datetime
from dataclasses import dataclass, field

from auction_gym.envs.rtb_env import RTBAuctionEnv
from auction_gym.core.agent import BaseAgent, AgentMeta
from auction_gym.core.valuation import BaseValuation, ValuationMeta
from auction_gym.core.utility import BaseUtility, UtilityMeta
from auction_gym.core.auction import AuctionMechanism, AuctionMechanismMeta

# Import concrete classes to trigger metaclass registration
from auction_gym.core.agent import RandomAgent, LinearAgent
from auction_gym.core.valuation import DeterministicValuation, UniformValuation, NormalValuation, GammaValuation
from auction_gym.core.utility import LinearUtility, RiskAverseUtility
from auction_gym.core.auction import FirstPriceAuction, SecondPriceAuction


@dataclass
class AgentConfig:
    """Lightweight, serializable configuration for agents."""
    agent_id: int
    agent_type: str 
    action_space: spaces.Space
    valuation_type: str 
    utility_type: str 
    is_trainable: bool = True
    # Use dictionaries for flexible parameters
    agent_params: dict = field(default_factory=dict)
    valuation_params: dict = field(default_factory=dict)
    utility_params: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default parameters if not provided."""
        # Validate types using metaclass registries
        available_agent_types = AgentMeta.get_available_agent_types()
        available_valuation_types = ValuationMeta.get_available_valuation_types()
        available_utility_types = UtilityMeta.get_available_utility_types()
        
        if self.agent_type not in available_agent_types:
            raise ValueError(f"Unsupported agent type: {self.agent_type}. Available types: {available_agent_types}")
        if self.valuation_type not in available_valuation_types:
            raise ValueError(f"Unsupported valuation type: {self.valuation_type}. Available types: {available_valuation_types}")
        if self.utility_type not in available_utility_types:
            raise ValueError(f"Unsupported utility type: {self.utility_type}. Available types: {available_utility_types}")
        
        # Provide defaults for required parameters
        if self.valuation_type == "deterministic" and "value" not in self.valuation_params:
            self.valuation_params["value"] = 0.5  # Default valuation

    @classmethod
    def create_agents(
        cls,
        num_agents: int,
        agent_type: str,
        valuation_type: str,
        utility_type: str,
        action_space: spaces.Space,
        is_trainable: bool = True,
        **params
    ) -> List['AgentConfig']:
        """
        Factory method for creating identical agent configs.
        
        Args:
            num_agents: Number of identical agents to create
            agent_type: Type for all agents
            valuation_type: Valuation type for all agents
            utility_type: Utility type for all agents
            action_space: Action space for the agents
            is_trainable: Whether all agents are trainable
            **params: Parameters for all agents (will be split by category)
        
        Returns:
            List of identical AgentConfig objects
        """
        # Split parameters by category
        agent_params = {k: v for k, v in params.items() 
                      if k.startswith('agent_') or k in ['lambda_param']}
        valuation_params = {k: v for k, v in params.items() 
                          if k.startswith('valuation_') or k in ['value', 'low', 'high', 'mean', 'std', 'shape', 'scale']}
        utility_params = {k: v for k, v in params.items() 
                        if k.startswith('utility_')}
        
        configs = []
        for i in range(num_agents):
            config = cls(
                agent_id=i,
                agent_type=agent_type,
                action_space=action_space,
                valuation_type=valuation_type,
                utility_type=utility_type,
                is_trainable=is_trainable,
                agent_params=agent_params.copy(),  # Copy to avoid shared references
                valuation_params=valuation_params.copy(),
                utility_params=utility_params.copy()
            )
            configs.append(config)
        return configs
    
    @classmethod
    def create_mixed_agents(
        cls,
        *agent_groups: Dict[str, Any]
    ) -> List['AgentConfig']:
        """
        Create mixed agent configurations using create_agents for each group.
        
        Args:
            *agent_groups: Variable number of dictionaries, each specifying:
                - num_agents: Number of agents in this group
                - All other parameters for create_agents
        
        Returns:
            List of AgentConfig objects from all groups
        
        Example:
            create_mixed_agents(
                {"num_agents": 2, "agent_type": "linear", "is_trainable": True},
                {"num_agents": 1, "agent_type": "random", "is_trainable": False}
            )
        """
        all_configs = []
        current_agent_id = 0
        
        for group in agent_groups:
            group = group.copy()  # Don't modify the original
            num_agents = group.pop('num_agents')
            
            # Create agents for this group
            group_configs = cls.create_agents(num_agents=num_agents, **group)
            
            # Update agent IDs to be sequential across all groups
            for config in group_configs:
                config.agent_id = current_agent_id
                current_agent_id += 1
            
            all_configs.extend(group_configs)
        
        return all_configs
    
    @classmethod
    def create_from_specs(cls, agent_specs: List[Dict[str, Any]]) -> List['AgentConfig']:
        """
        Create configs from individual specifications (for maximum flexibility).
        
        Args:
            agent_specs: List of dictionaries with individual agent specifications
        
        Returns:
            List of AgentConfig objects
        """
        configs = []
        for spec in agent_specs:
            agent_id = spec.pop('agent_id')
            config = cls(agent_id=agent_id, **spec)
            configs.append(config)
        return configs
    
    
    


@dataclass
class AuctionExperimentConfig:
    """Configuration for auction experiments."""
    experiment_name: str
    algorithm: str = "PPO"
    max_steps: int = 100
    auction_type: str = "second_price"
    num_agents: int = None  # Will be inferred from agent_configs if not provided
    agent_configs: List[AgentConfig] = field(default_factory=list)
    
    # Training parameters
    num_iterations: int = 100
    checkpoint_freq: int = 10
    verbose: int = 1
    
    # Algorithm-specific parameters
    algorithm_params: dict = field(default_factory=dict)
    
    # Environment parameters
    env_params: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and set derived parameters."""
        # Infer num_agents from agent_configs if not provided
        if self.num_agents is None:
            if self.agent_configs:
                self.num_agents = len(self.agent_configs)
            else:
                self.num_agents = 2  # Default
        
        # Validate that agent_configs matches num_agents if both are provided
        if self.agent_configs and len(self.agent_configs) != self.num_agents:
            raise ValueError(f"Number of agent configs ({len(self.agent_configs)}) doesn't match num_agents ({self.num_agents})")
        
        # Validate algorithm
        if self.algorithm.upper() not in ["PPO", "SAC", "DQN"]:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Validate auction type
        available_auction_types = AuctionMechanismMeta.get_available_mechanism_types()
        if self.auction_type not in available_auction_types:
            raise ValueError(f"Unsupported auction type: {self.auction_type}. Available types: {available_auction_types}")
    
    @classmethod
    def create_simple_experiment(
        cls,
        experiment_name: str,
        num_agents: int = 2,
        trainable_agents: List[int] = None,
        agent_type: str = "linear",
        valuation_type: str = "deterministic",
        utility_type: str = "linear",
        algorithm: str = "PPO",
        **kwargs
    ) -> 'AuctionExperimentConfig':
        """
        Create a simple experiment configuration with identical agents.
        
        Args:
            experiment_name: Name of the experiment
            num_agents: Number of agents
            trainable_agents: List of agent IDs that should be trainable
            agent_type: Type for all agents
            valuation_type: Valuation type for all agents
            utility_type: Utility type for all agents
            algorithm: RL algorithm to use
            **kwargs: Additional parameters
        """
        if trainable_agents is None:
            trainable_agents = [0]
        
        # Create a default action space (can be overridden in kwargs)
        default_action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        action_space = kwargs.pop('action_space', default_action_space)
        
        # Create trainable and fixed agent groups
        agent_configs = AgentConfig.create_mixed_agents(
            {
                "num_agents": len(trainable_agents),
                "agent_type": agent_type,
                "valuation_type": valuation_type,
                "utility_type": utility_type,
                "action_space": action_space,
                "is_trainable": True,
                **{k: v for k, v in kwargs.items() if not k.startswith('algorithm_')}
            },
            {
                "num_agents": num_agents - len(trainable_agents),
                "agent_type": agent_type,
                "valuation_type": valuation_type,
                "utility_type": utility_type,
                "action_space": action_space,
                "is_trainable": False,
                **{k: v for k, v in kwargs.items() if not k.startswith('algorithm_')}
            }
        ) if num_agents > len(trainable_agents) else AgentConfig.create_agents(
            num_agents=num_agents,
            agent_type=agent_type,
            valuation_type=valuation_type,
            utility_type=utility_type,
            action_space=action_space,
            is_trainable=True,
            **{k: v for k, v in kwargs.items() if not k.startswith('algorithm_')}
        )
        
        # Reorder configs to match trainable_agents order
        if num_agents > len(trainable_agents):
            reordered_configs = [None] * num_agents
            trainable_idx = 0
            fixed_idx = len(trainable_agents)
            
            for i in range(num_agents):
                if i in trainable_agents:
                    config = agent_configs[trainable_idx]
                    config.agent_id = i
                    reordered_configs[i] = config
                    trainable_idx += 1
                else:
                    config = agent_configs[fixed_idx]
                    config.agent_id = i
                    reordered_configs[i] = config
                    fixed_idx += 1
            agent_configs = reordered_configs
        
        # Extract algorithm parameters
        algorithm_params = {k[10:]: v for k, v in kwargs.items() if k.startswith('algorithm_')}
        
        return cls(
            experiment_name=experiment_name,
            num_agents=num_agents,
            agent_configs=agent_configs,
            algorithm=algorithm,
            algorithm_params=algorithm_params,
            **{k: v for k, v in kwargs.items() 
               if not k.startswith('algorithm_') 
               and k not in ['agent_type', 'valuation_type', 'utility_type', 'lambda_param', 'value', 'low', 'high', 'mean', 'std', 'shape', 'scale']}
        )
    
    @classmethod
    def create_mixed_experiment(
        cls,
        experiment_name: str,
        agent_groups: List[Dict[str, Any]],
        algorithm: str = "PPO",
        **kwargs
    ) -> 'AuctionExperimentConfig':
        """
        Create an experiment configuration with mixed agent types.
        
        Args:
            experiment_name: Name of the experiment
            agent_groups: List of agent group specifications
            algorithm: RL algorithm to use
            **kwargs: Additional parameters
        """
        agent_configs = AgentConfig.create_mixed_agents(*agent_groups)
        
        # Extract algorithm parameters
        algorithm_params = {k[10:]: v for k, v in kwargs.items() if k.startswith('algorithm_')}
        
        return cls(
            experiment_name=experiment_name,
            agent_configs=agent_configs,
            algorithm=algorithm,
            algorithm_params=algorithm_params,
            **{k: v for k, v in kwargs.items() if not k.startswith('algorithm_')}
        )
    
    @classmethod
    def create_from_agent_configs(
        cls,
        experiment_name: str,
        agent_configs: List[AgentConfig],
        algorithm: str = "PPO",
        **kwargs
    ) -> 'AuctionExperimentConfig':
        """
        Create an experiment configuration from existing agent configs.
        
        Args:
            experiment_name: Name of the experiment
            agent_configs: List of AgentConfig objects
            algorithm: RL algorithm to use
            **kwargs: Additional parameters
        """
        # Extract algorithm parameters
        algorithm_params = {k[10:]: v for k, v in kwargs.items() if k.startswith('algorithm_')}
        
        return cls(
            experiment_name=experiment_name,
            agent_configs=agent_configs,
            algorithm=algorithm,
            algorithm_params=algorithm_params,
            **{k: v for k, v in kwargs.items() if not k.startswith('algorithm_')}
        )
    
    def get_learning_agent_ids(self) -> List[int]:
        """Get list of trainable agent IDs."""
        return [config.agent_id for config in self.agent_configs if config.is_trainable]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_name": self.experiment_name,
            "algorithm": self.algorithm,
            "max_steps": self.max_steps,
            "auction_type": self.auction_type,
            "num_agents": self.num_agents,
            "num_iterations": self.num_iterations,
            "checkpoint_freq": self.checkpoint_freq,
            "verbose": self.verbose,
            "algorithm_params": self.algorithm_params,
            "env_params": self.env_params,
            "agent_configs": [
                {
                    "agent_id": config.agent_id,
                    "agent_type": config.agent_type,
                    "action_space": {
                        "type": "Box",
                        "low": config.action_space.low.tolist() if hasattr(config.action_space, 'low') else 0.0,
                        "high": config.action_space.high.tolist() if hasattr(config.action_space, 'high') else 1.0,
                        "shape": config.action_space.shape,
                        "dtype": str(config.action_space.dtype)
                    },
                    "valuation_type": config.valuation_type,
                    "utility_type": config.utility_type,
                    "is_trainable": config.is_trainable,
                    "agent_params": config.agent_params,
                    "valuation_params": config.valuation_params,
                    "utility_params": config.utility_params,
                }
                for config in self.agent_configs
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuctionExperimentConfig':
        """Create from dictionary (for deserialization)."""
        agent_configs_data = data.pop("agent_configs", [])
        agent_configs = []
        
        for config_data in agent_configs_data:
            # Reconstruct action_space from serialized data
            action_space_data = config_data.get("action_space", {})
            if action_space_data.get("type") == "Box":
                import numpy as np
                dtype_map = {"float32": np.float32, "float64": np.float64, "int32": np.int32, "int64": np.int64}
                dtype = dtype_map.get(action_space_data.get("dtype", "float32"), np.float32)
                action_space = spaces.Box(
                    low=np.array(action_space_data.get("low", 0.0), dtype=dtype),
                    high=np.array(action_space_data.get("high", 1.0), dtype=dtype),
                    shape=tuple(action_space_data.get("shape", (1,))),
                    dtype=dtype
                )
            else:
                # Default fallback
                action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            
            agent_config = AgentConfig(
                agent_id=config_data["agent_id"],
                agent_type=config_data["agent_type"],
                action_space=action_space,
                valuation_type=config_data["valuation_type"],
                utility_type=config_data["utility_type"],
                is_trainable=config_data["is_trainable"],
                agent_params=config_data.get("agent_params", {}),
                valuation_params=config_data.get("valuation_params", {}),
                utility_params=config_data.get("utility_params", {}),
            )
            agent_configs.append(agent_config)
        
        return cls(agent_configs=agent_configs, **data)


class AuctionExperiment:
    """
    Experiment class for training agents in auction environments using RLlib.
    """
    
    def __init__(self, config: AuctionExperimentConfig):
        """
        Initialize the experiment from configuration.
        
        Args:
            config: AuctionExperimentConfig object
        """
        self.config = config
        self.experiment_name = config.experiment_name
        self.num_agents = config.num_agents
        self.algorithm = config.algorithm.upper()
        self.max_steps = config.max_steps
        self.auction_type = config.auction_type
        self.agent_configs = config.agent_configs
        self.learning_agent_ids = config.get_learning_agent_ids()
        
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Register environment
        register_env("auction_env", self._env_creator)
        
        # Results storage
        self.results = None
        self.checkpoint_path = None
    
    def _create_agent_configs(self) -> List[AgentConfig]:
        """Return the agent configurations from the experiment config."""
        return self.agent_configs
    
    def _env_creator(self, env_config: Dict[str, Any]) -> RTBAuctionEnv:
        """
        Creates an instance of the auction environment.
        
        Args:
            env_config: Configuration for the environment.
        
        Returns:
            An instance of RTBAuctionEnv.
        """
        # Create agents from configs
        agents = []
        for agent_config in self.agent_configs:
            # Get classes using metaclasses
            agent_class = AgentMeta.get_agent_class(agent_config.agent_type)
            valuation_class = ValuationMeta.get_valuation_class(agent_config.valuation_type) 
            utility_class = UtilityMeta.get_utility_class(agent_config.utility_type)
            action_space = agent_config.action_space
            # Create instances
            valuation = valuation_class(**agent_config.valuation_params)
            utility = utility_class(**agent_config.utility_params)
            
            
            # Create agent
            agent = agent_class(
                action_space=action_space,
                valuation=valuation,
                utility=utility,
                is_trainable=agent_config.is_trainable,
                **agent_config.agent_params
            )
            agents.append(agent)
        
        # Create auction mechanism
        auction_mechanism = AuctionMechanismMeta.get_mechanism_class(self.auction_type)()
        
        return RTBAuctionEnv(
            agents=agents,
            auction_mechanism=auction_mechanism
        )
    
    def _get_algorithm_config(self):
        """Get algorithm configuration based on selected algorithm."""
        if self.algorithm == "PPO":
            config = (
                PPOConfig()
                .environment("auction_env")
                .framework("torch")
                .training(
                    train_batch_size=self.config.algorithm_params.get("train_batch_size", 4000),
                    lr=self.config.algorithm_params.get("lr", 0.0003),
                    gamma=self.config.algorithm_params.get("gamma", 0.99),
                    lambda_=self.config.algorithm_params.get("lambda_", 0.95),
                    entropy_coeff=self.config.algorithm_params.get("entropy_coeff", 0.01),
                    num_epochs=self.config.algorithm_params.get("num_epochs", 10),
                )
                .env_runners(
                    num_env_runners=self.config.algorithm_params.get("num_env_runners", 2),
                )
                .resources(num_gpus=self.config.algorithm_params.get("num_gpus", 0))
            )
        elif self.algorithm == "SAC":
            config = (
                SACConfig()
                .environment("auction_env")
                .framework("torch")
                .training(
                    train_batch_size=self.config.algorithm_params.get("train_batch_size", 4000),
                    lr=self.config.algorithm_params.get("lr", 0.0003),
                    gamma=self.config.algorithm_params.get("gamma", 0.99),
                    lambda_=self.config.algorithm_params.get("lambda_", 0.95),
                    entropy_coeff=self.config.algorithm_params.get("entropy_coeff", 0.01),
                    num_epochs=self.config.algorithm_params.get("num_epochs", 10),
                )
                .env_runners(
                    num_env_runners=self.config.algorithm_params.get("num_env_runners", 2),
                )
                .resources(num_gpus=self.config.algorithm_params.get("num_gpus", 0))
            )
        elif self.algorithm == "DQN":
            config = (
                DQNConfig()
                .environment("auction_env")
                .framework("torch")
                .training(
                    train_batch_size=self.config.algorithm_params.get("train_batch_size", 4000),
                    lr=self.config.algorithm_params.get("lr", 0.0003),
                    gamma=self.config.algorithm_params.get("gamma", 0.99),
                    lambda_=self.config.algorithm_params.get("lambda_", 0.95),
                    entropy_coeff=self.config.algorithm_params.get("entropy_coeff", 0.01),
                    num_epochs=self.config.algorithm_params.get("num_epochs", 10),
                )
                .env_runners(
                    num_env_runners=self.config.algorithm_params.get("num_env_runners", 2),
                )
                .resources(num_gpus=self.config.algorithm_params.get("num_gpus", 0))
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Add multi-agent configuration with proper policy mapping
        def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
            """Map agent IDs to policy IDs."""
            return agent_id
        
        # Determine which policies should be trained
        trainable_policy_ids = [f"agent_{i}" for i in self.learning_agent_ids]
        
        config = config.multi_agent(
            policies={
                f"agent_{i}": (
                    None,  # Use default policy class for the algorithm
                    self.config.env_params.get("observation_space", None),
                    self.config.env_params.get("action_space", None),
                    {"agent_id": i}
                )
                for i in range(self.num_agents)
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=trainable_policy_ids,
        )
        
        return config
    
    def run(self, **override_params):
        """
        Run the experiment.
        
        Args:
            **override_params: Parameters to override from config
        """
        num_iterations = override_params.get("num_iterations", self.config.num_iterations)
        checkpoint_freq = override_params.get("checkpoint_freq", self.config.checkpoint_freq)
        verbose = override_params.get("verbose", self.config.verbose)
        
        # Get algorithm configuration
        algorithm_config = self._get_algorithm_config()
        
        # Build the algorithm
        algorithm = algorithm_config.build()
        
        # Train the algorithm
        training_results = []
        for i in range(num_iterations):
            result = algorithm.train()
            training_results.append(result)
            
            # Check if we should checkpoint
            if checkpoint_freq > 0 and (i + 1) % checkpoint_freq == 0:
                checkpoint_path = algorithm.save()
                print(f"Checkpointed at iteration {i+1}: {checkpoint_path}")
            
            if verbose >= 1:
                print(f"Iteration {i+1}: {result}")
        
        # Final checkpoint
        if checkpoint_freq > 0:
            checkpoint_path = algorithm.save()
            print(f"Final checkpoint: {checkpoint_path}")
        
        results = training_results
        
        self.results = results
        self.checkpoint_path = algorithm.get_checkpoint() if hasattr(algorithm, 'get_checkpoint') else None
        print(f"Experiment finished. Completed {num_iterations} iterations.")
        return results
    
    def save_experiment(self, filepath: str):
        """Save experiment configuration and results."""
        experiment_data = self.config.to_dict()
        experiment_data.update({
            "checkpoint_path": self.checkpoint_path,
            "timestamp": datetime.now().isoformat()
        })
        
        with open(filepath, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        print(f"Experiment saved to: {filepath}")
    
    @classmethod
    def load_experiment(cls, filepath: str):
        """Load experiment from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        checkpoint_path = data.pop("checkpoint_path", None)
        data.pop("timestamp", None)  # Remove timestamp
        
        config = AuctionExperimentConfig.from_dict(data)
        experiment = cls(config)
        experiment.checkpoint_path = checkpoint_path
        
        return experiment


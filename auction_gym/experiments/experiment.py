import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import PolicyID
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import json
import os
from datetime import datetime
import numpy as np
from gymnasium import spaces

from auction_gym.core.policy import BiddingPolicy, BiddingAgentConfig
from auction_gym.core.mechanism import AuctionMechanism, SecondPriceAuction, FirstPriceAuction
from auction_gym.envs.auction_env import Auction


@dataclass
class ExperimentConfig:
    """Configuration dataclass for auction experiments."""
    
    # Experiment metadata
    experiment_name: str = "auction_experiment"
    description: str = ""
    
    # Environment configuration
    num_agents: int = 2
    auction_type: str = "second_price"  # "second_price" or "first_price"
    max_steps: int = 100
    env_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    algorithm: str = "PPO"  # "PPO", "DQN", "A3C", "IMPALA"
    num_iterations: int = 100
    checkpoint_freq: int = 10
    evaluation_freq: int = 5
    evaluation_duration: int = 10
    
    # Algorithm-specific parameters
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    
    # Agent configurations - each agent config will be used to create a BiddingAgent instance
    agent_configs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Logging and output
    log_dir: str = "./results"
    verbose: int = 1
    save_checkpoints: bool = True
    
    # Ray configuration
    num_cpus: int = 4
    num_gpus: int = 0
    local_mode: bool = False
    
    def __post_init__(self):
        """Validate and set default agent configurations if not provided."""
        if not self.agent_configs:
            # Create default agent configurations
            self.agent_configs = []
            for i in range(self.num_agents):
                agent_config = {
                    "agent_id": i,
                    "valuation_type": "deterministic",
                    "utility_type": "linear", 
                    "model_type": "linear",
                    "is_trainable": True,
                    "valuation_config": {"value": 0.5 + i * 0.1},  # Different values for each agent
                    "utility_config": {},
                    "model_config": {}
                }
                self.agent_configs.append(agent_config)


class AuctionExperiment:
    """Main experiment class for training auction agents."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the experiment with configuration.
        
        Args:
            config: ExperimentConfig object containing all experiment parameters
        """
        self.config = config
        self.agents = []
        self.env = None
        self.trainer = None
        self.results = {}
        
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(
                num_cpus=config.num_cpus,
                num_gpus=config.num_gpus,
                local_mode=config.local_mode
            )
    
    def build_agents(self) -> List[BiddingPolicy]:
        """Build BiddingAgent instances based on the experiment config."""
        agents = []
        
        # Create observation and action spaces
        observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        for agent_config in self.config.agent_configs:
            # Create BiddingAgentConfig
            bidding_config = BiddingAgentConfig(
                valuation_type=agent_config.get("valuation_type", "deterministic"),
                utility_type=agent_config.get("utility_type", "linear"),
                model_type=agent_config.get("model_type", "linear"),
                is_trainable=agent_config.get("is_trainable", True),
                valuation_config=agent_config.get("valuation_config", {"value": 0.5}),
                utility_config=agent_config.get("utility_config", {}),
                model_config=agent_config.get("model_config", {})
            )
            
            # Create BiddingAgent instance
            agent = BiddingPolicy(observation_space, action_space, bidding_config)
            agents.append(agent)
        
        self.agents = agents
        return agents
    
    def build_environment(self) -> Auction:
        """Build the auction environment with the configured agents."""
        if not self.agents:
            self.build_agents()
        
        # Create auction mechanism
        if self.config.auction_type == "second_price":
            auction_mechanism = SecondPriceAuction()
        elif self.config.auction_type == "first_price":
            auction_mechanism = FirstPriceAuction()
        else:
            raise ValueError(f"Unknown auction type: {self.config.auction_type}")
        
        # Create environment with the actual BiddingAgent instances
        env = Auction(
            agents=self.agents,
            mechanism=auction_mechanism
        )
        
        # Set max_steps if provided
        if hasattr(env, 'max_steps'):
            env.max_steps = self.config.max_steps
        
        self.env = env
        return env
    
    def create_policies(self) -> Dict[PolicyID, Policy]:
        """Create policies dictionary using the actual BiddingAgent instances.
        
        Since BiddingAgent inherits from TorchPolicy, we can use the agent instances
        directly as policies.
        """
        policies = {}
        
        for i, agent in enumerate(self.agents):
            agent_id = f"agent_{i}"
            policies[agent_id] = agent
        
        return policies
    
    def get_algorithm_config(self):
        """Get the RLlib algorithm configuration based on the experiment config."""
        base_config = {
            "env": Auction,
            "env_config": {
                "agents": self.agents,  # Pass the actual agent instances
                "auction_mechanism": self.config.auction_type,
                "max_steps": self.config.max_steps,
                **self.config.env_config
            },
            "multiagent": {
                "policies": self.create_policies(),
                "policy_mapping_fn": lambda agent_id: agent_id,
            },
            "num_workers": 1,
            "framework": "torch",
            **self.config.algorithm_params
        }
        
        # Create algorithm-specific config
        if self.config.algorithm == "PPO":
            config = PPOConfig().from_dict(base_config)
        elif self.config.algorithm == "DQN":
            config = DQNConfig().from_dict(base_config)
        elif self.config.algorithm == "A3C":
            config = A3CConfig().from_dict(base_config)
        elif self.config.algorithm == "IMPALA":
            config = ImpalaConfig().from_dict(base_config)
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        return config
    
    def train(self) -> Dict[str, Any]:
        """Train the agents using RLlib."""
        if self.config.verbose >= 1:
            print(f"Starting training with {self.config.algorithm} algorithm")
            print(f"Number of agents: {len(self.agents)}")
            print(f"Auction type: {self.config.auction_type}")
        
        # Build agents and environment
        self.build_environment()
        
        # Get algorithm configuration
        config = self.get_algorithm_config()
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(
            self.config.log_dir, 
            f"{self.config.experiment_name}_{timestamp}"
        )
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Run training
        results = tune.run(
            self.config.algorithm,
            config=config.to_dict(),
            stop={"training_iteration": self.config.num_iterations},
            checkpoint_freq=self.config.checkpoint_freq,
            local_dir=experiment_dir,
            verbose=self.config.verbose
        )
        
        self.results = results
        return results
    
    def evaluate(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate the trained agents."""
        if checkpoint_path is None and self.results:
            # Use the best checkpoint from training
            checkpoint_path = self.results.get_best_checkpoint(
                metric="episode_reward_mean", 
                mode="max"
            )
        
        if checkpoint_path is None:
            raise ValueError("No checkpoint path provided and no training results available")
        
        # Build environment for evaluation
        self.build_environment()
        
        # Load the trained agent
        config = self.get_algorithm_config()
        trainer = config.build()
        trainer.restore(checkpoint_path)
        
        # Run evaluation episodes
        eval_results = []
        for episode in range(self.config.evaluation_duration):
            obs, info = self.env.reset()
            episode_reward = {agent_id: 0.0 for agent_id in obs.keys()}
            episode_length = 0
            
            while True:
                actions = trainer.compute_actions(obs)
                obs, rewards, terminated, truncated, info = self.env.step(actions)
                
                # Accumulate rewards
                for agent_id in rewards:
                    episode_reward[agent_id] += rewards[agent_id]
                
                episode_length += 1
                
                if terminated.get("__all__", False) or truncated.get("__all__", False):
                    break
            
            eval_results.append({
                "episode": episode,
                "length": episode_length,
                "rewards": episode_reward
            })
        
        trainer.stop()
        
        # Calculate evaluation metrics
        avg_rewards = {}
        for agent_id in self.env.agents:
            agent_rewards = [ep["rewards"][agent_id] for ep in eval_results]
            avg_rewards[agent_id] = np.mean(agent_rewards)
        
        evaluation_summary = {
            "checkpoint_path": checkpoint_path,
            "num_episodes": len(eval_results),
            "average_rewards": avg_rewards,
            "episode_details": eval_results
        }
        
        return evaluation_summary
    
    def save_experiment_config(self, filepath: Optional[str] = None):
        """Save the experiment configuration to a JSON file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                self.config.log_dir,
                f"{self.config.experiment_name}_{timestamp}_config.json"
            )
        
        # Convert config to dict for JSON serialization
        config_dict = {
            "experiment_name": self.config.experiment_name,
            "description": self.config.description,
            "num_agents": self.config.num_agents,
            "auction_type": self.config.auction_type,
            "max_steps": self.config.max_steps,
            "algorithm": self.config.algorithm,
            "num_iterations": self.config.num_iterations,
            "checkpoint_freq": self.config.checkpoint_freq,
            "evaluation_freq": self.config.evaluation_freq,
            "evaluation_duration": self.config.evaluation_duration,
            "algorithm_params": self.config.algorithm_params,
            "agent_configs": self.config.agent_configs,
            "env_config": self.config.env_config,
            "timestamp": datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        return filepath
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment: training + evaluation."""
        if self.config.verbose >= 1:
            print("=" * 50)
            print(f"Starting experiment: {self.config.experiment_name}")
            print("=" * 50)
        
        # Save experiment configuration
        config_path = self.save_experiment_config()
        if self.config.verbose >= 1:
            print(f"Saved experiment config to: {config_path}")
        
        # Train agents
        training_results = self.train()
        
        # Evaluate agents
        evaluation_results = self.evaluate()
        
        # Combine results
        full_results = {
            "experiment_name": self.config.experiment_name,
            "config_path": config_path,
            "training_results": training_results,
            "evaluation_results": evaluation_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save full results
        results_path = os.path.join(
            self.config.log_dir,
            f"{self.config.experiment_name}_full_results.json"
        )
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        if self.config.verbose >= 1:
            print(f"Saved full results to: {results_path}")
            print("=" * 50)
            print("Experiment completed!")
            print("=" * 50)
        
        return full_results
    
    def cleanup(self):
        """Clean up resources."""
        if self.trainer:
            self.trainer.stop()
        if ray.is_initialized():
            ray.shutdown()


# Example usage function
def run_example_experiment():
    """Example of how to use the AuctionExperiment class."""
    
    # Create experiment configuration
    config = ExperimentConfig(
        experiment_name="example_auction_experiment",
        description="Example experiment with 2 agents using PPO",
        num_agents=2,
        auction_type="second_price",
        algorithm="PPO",
        num_iterations=50,
        checkpoint_freq=10,
        evaluation_duration=5,
        algorithm_params={
            "lr": 0.001,
            "train_batch_size": 1000,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 10,
        },
        agent_configs=[
            {
                "agent_id": 0,
                "valuation_type": "deterministic",
                "utility_type": "linear",
                "model_type": "linear",
                "is_trainable": True,
                "valuation_config": {"value": 0.6},
                "utility_config": {},
                "model_config": {}
            },
            {
                "agent_id": 1,
                "valuation_type": "deterministic", 
                "utility_type": "linear",
                "model_type": "linear",
                "is_trainable": True,
                "valuation_config": {"value": 0.7},
                "utility_config": {},
                "model_config": {}
            }
        ]
    )
    
    # Create and run experiment
    experiment = AuctionExperiment(config)
    try:
        results = experiment.run_full_experiment()
        print("Experiment completed successfully!")
        return results
    finally:
        experiment.cleanup()


if __name__ == "__main__":
    run_example_experiment()

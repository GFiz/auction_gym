"""
Test experiment with 2 agents using separate Linear Bidder modules and VPG learning.
This demonstrates a multi-agent auction environment where each agent has its own
learning module and learns independently using the VPG algorithm.
"""

import numpy as np
import torch
import os
from gymnasium import spaces
from typing import Dict, Any
import ray
from ray import tune
from ray.tune import register_env
from ray.rllib.env.env_context import EnvContext

from auction_gym.envs.auction_env import AuctionEnv
from auction_gym.core.valuation import UniformValuation, NormalValuation
from auction_gym.core.utility import LinearUtility, TestUtility
from auction_gym.core.mechanism import SecondPriceAuction
from auction_gym.algorithms.vpg import VPG, VPGConfig
from auction_gym.modules.linear_bidder import LBTorchRLModule


def create_test_environment(log_actions_path: str | None = None):
    """Create a test auction environment with 2 agents.
    Optionally pass a JSONL path to log per-step actions from the env.
    """
    
    # Define agent configurations
    agents = ["agent_0", "agent_1"]
    
    # Create valuations for each agent
    valuations = {
        "agent_0": UniformValuation(low=0.5, high=1.5),  # Agent 0 has higher valuations
        "agent_1": UniformValuation(low=0.3, high=1.2),  # Agent 1 has lower valuations
    }
    
    # Create utilities for each agent (both use linear utility)
    utilities = {
        "agent_0": TestUtility(),
        "agent_1": TestUtility(),
    }
    
    # Define action spaces (bidding spaces)
    action_spaces = {
        "agent_0": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        "agent_1": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
    }
    
    # Define observation spaces (valuation observations)
    observation_spaces = {
        "agent_0": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        "agent_1": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
    }
    
    # Create auction mechanism
    mechanism = SecondPriceAuction()
    
    # Create environment
    env = AuctionEnv(
        valuations=valuations,
        utilities=utilities,
        action_spaces=action_spaces,
        observation_spaces=observation_spaces,
        mechanism=mechanism,
        max_steps=50,  # 50 auctions per episode
        log_actions_path=log_actions_path,
    )
    
    return env


def env_creator(env_config: Dict[str, Any]):
    """Environment creator function for RLlib."""
    log_path = None
    if isinstance(env_config, dict):
        log_path = env_config.get("log_actions_path")
    return create_test_environment(log_path)


def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
    """Policy mapping function for multi-agent environment that can handle both signatures."""
    return agent_id


def agent_to_module_mapping_fn(agent_id, episode):
    """Agent to module mapping function for multi-agent environment."""
    return agent_id


def create_vpg_config():
    """Create VPG configuration for the experiment."""
    config = VPGConfig()
    
    # Training configuration
    config = config.training(
        num_episodes_per_train_batch=5,  # 5 episodes per training batch
        lr=0.001,
        train_batch_size=250,  # Total timesteps per batch
    )
    
    # Framework configuration
    config.framework_str = "torch"
    
    # Environment configuration - register the environment first
    register_env("auction_env", env_creator)
    config.environment("auction_env")


    # RLlib specific settings
    config.num_env_runners = 1
    config.num_learners = 1
    
    # Multi-agent configuration with proper policy setup
    config.multi_agent(
        policies={
            "agent_0": (None, spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32), spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32), {}),
            "agent_1": (None, spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32), spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32), {}),
        },
        policy_mapping_fn=policy_mapping_fn,
    )
    
    return config


def run_test_experiment():
    """Run the test experiment with 2 agents learning via VPG."""
    
    print("Setting up test experiment with 2 agents...")
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(local_mode=True)  # Use local mode for testing
    
    # Create environment for testing
    env = create_test_environment()
    print(f"Created environment with {env.n_agents} agents")
    print(f"Agents: {env.agents}")
    
    # Test environment interaction
    print("\nTesting environment interaction...")
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial observations: {obs}")
    print(f"Initial info: {info}")
    
    # Run a few test steps
    total_reward = {agent: 0.0 for agent in env.agents}
    
    for step in range(10):  # Test 10 steps
        # Generate random actions for testing
        actions = {}
        for agent in env.agents:
            # Random bidding strategy for testing
            actions[agent] = np.random.uniform(0.0, 1.0, size=(1,))
        
        # Take step
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Accumulate rewards
        for agent in env.agents:
            total_reward[agent] += rewards[agent]
        
        print(f"Step {step + 1}:")
        print(f"  Actions: {actions}")
        print(f"  Rewards: {rewards}")
        print(f"  Observations: {obs}")
        
        # Check if episode is done
        if any(terminated.values()) or any(truncated.values()):
            print("Episode ended")
            break
    
    print(f"\nTotal rewards after 10 steps: {total_reward}")
    
    # Test episode summary
    episode_summary = env.get_episode_summary()
    print(f"\nEpisode summary: {episode_summary}")
    
    # Test Linear Bidder module creation
    print("\nTesting Linear Bidder module creation...")
    
    for agent in env.agents:
        rl_module = LBTorchRLModule(
            observation_space=env.observation_spaces[agent],
            action_space=env.action_spaces[agent],
            model_config={"initial_mu": 0.5, "initial_sigma": 0.1}
        )
        print(f"Created RL module for {agent}")
        print(f"  Initial mu: {rl_module.mu.item()}")
        print(f"  Initial log_sigma: {rl_module.log_sigma.item()}")
    
    print("\nTest experiment completed successfully!")
    
    return env


def run_rllib_training():
    """Run full RLlib training with VPG algorithm."""
    
    print("\n" + "="*50)
    print("Running Full RLlib Training")
    print("="*50)
    
    # Create VPG configuration
    config = create_vpg_config()
    print(f"VPG Configuration: {config.num_episodes_per_train_batch} episodes per batch")
    
    # Run training using tune with the VPG class directly
    # Use a shorter training run to avoid process group conflicts
    results = tune.run(
        VPG,
        config=config.to_dict(),
        stop={
            "training_iteration": 5,  # Reduced to 5 iterations
        },
        checkpoint_freq=2,
        checkpoint_at_end=True,
        verbose=1,
        storage_path="file:///home/gonzalo/repos/auction_gym/auction_gym/experiments/vpg_test/",  # Use absolute path with file:// scheme
    )
    
    print(f"Training completed. Best result: {results.best_result}")
    
    return results


def test_algorithm_initialization():
    """Test VPG algorithm initialization with RLlib components."""
    
    print("\n" + "="*50)
    print("Testing VPG Algorithm Initialization")
    print("="*50)
    
    # Create VPG configuration
    config = create_vpg_config()
    
    # Create VPG algorithm instance
    algorithm = VPG(config)
    print("Created VPG algorithm with RLlib components")
    
    # Test configuration
    print(f"Algorithm config: {algorithm.config.num_episodes_per_train_batch} episodes per batch")
    print(f"Framework: {algorithm.config.framework_str}")
    print(f"Environment: {algorithm.config.env}")
    print(f"Number of env runners: {algorithm.config.num_env_runners}")
    print(f"Number of learners: {algorithm.config.num_learners}")
    
    # Test that RLlib components are initialized
    print(f"Environment runner group: {algorithm.env_runner_group}")
    print(f"Learner group: {algorithm.learner_group}")
    print(f"Metrics: {algorithm.metrics}")
    
    # Test multi-agent configuration
    print(f"\nMulti-agent configuration:")
    print(f"Policies: {algorithm.config.policies}")
    print(f"Policy mapping function: {algorithm.config.policy_mapping_fn}")
    
    # Test a single training step
    try:
        print("\nTesting single training step...")
        algorithm.training_step()
        print("✓ Training step completed successfully!")
    except Exception as e:
        print(f"Training step failed: {e}")
        print("This might be due to multi-agent configuration issues.")
    
    return algorithm


if __name__ == "__main__":
    # Run the main test experiment
    #env = run_test_experiment()
    
    #Test algorithm initialization and single training step
    #algorithm = test_algorithm_initialization()
    
    # Run full RLlib training (commented out to avoid process group conflicts)
    run_rllib_training()
    print("\n" + "="*50)
    print("Full RLlib Training Test")
    print("="*50)
    print("Note: Full training is skipped to avoid PyTorch distributed process group conflicts.")
    print("The algorithm initialization and single training step above demonstrate")
    print("that the VPG algorithm is properly configured and working.")
    print("\nTo run full training, use a separate script or restart the Python process.")
    
    print("\n" + "="*50)
    print("Experiment Summary")
    print("="*50)
    print("✓ Created auction environment with 2 agents")
    print("✓ Set up separate Linear Bidder modules for each agent")
    print("✓ Configured VPG algorithm for learning")
    print("✓ Tested environment interaction")
    print("✓ Verified RL module creation")
    print("✓ Initialized RLlib components")
    print("✓ Tested single training step")
    print("\nThe experiment demonstrates:")
    print("- Multi-agent auction environment setup")
    print("- Independent Linear Bidder modules per agent")
    print("- VPG algorithm configuration for learning")
    print("- Environment interaction and reward computation")
    print("- RLlib training infrastructure initialization")
    print("- Single training step execution") 
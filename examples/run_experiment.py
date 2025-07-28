"""
Example script demonstrating how to use the AuctionExperiment class.
"""

import numpy as np
from auction_gym.experiments.experiment import AuctionExperiment, AuctionExperimentConfig


def run_single_agent_experiment():
    """Run a single-agent learning experiment."""
    print("=== Single Agent Learning Experiment ===\n")
    
    # Create experiment: agent 0 learns against fixed agent 1
    config = AuctionExperimentConfig.create_simple_experiment(
        experiment_name="single_agent_vs_fixed",
        num_agents=2,
        trainable_agents=[0],  # Only agent 0 learns
        agent_type="linear",
        valuation_type="deterministic",
        utility_type="linear",
        algorithm="PPO",
        max_steps=50,
        auction_type="second_price",
        num_iterations=50,  # Short training for demo
        checkpoint_freq=10,
        verbose=1,
        # Agent configuration
        lambda_param=0.8,  # Strategic bidder parameter
        value=0.6,  # Valuation
    )
    
    experiment = AuctionExperiment(config)
    
    # Run training
    results = experiment.run()
    
    # Save experiment
    experiment.save_experiment("results/single_agent_experiment.json")
    
    return experiment, results


def run_multi_agent_experiment():
    """Run a multi-agent learning experiment."""
    print("\n=== Multi-Agent Learning Experiment ===\n")
    
    # Create experiment: agents 0 and 2 learn against fixed agents 1 and 3
    config = AuctionExperimentConfig.create_simple_experiment(
        experiment_name="multi_agent_learning",
        num_agents=4,
        trainable_agents=[0, 2],  # Agents 0 and 2 learn
        agent_type="linear",
        valuation_type="deterministic",
        utility_type="linear",
        algorithm="PPO",
        max_steps=50,
        auction_type="second_price",
        num_iterations=50,  # Short training for demo
        checkpoint_freq=10,
        verbose=1,
        # Agent configuration
        lambda_param=0.8,
        value=0.6,
    )
    
    experiment = AuctionExperiment(config)
    
    # Run training
    results = experiment.run()
    
    # Save experiment
    experiment.save_experiment("results/multi_agent_experiment.json")
    
    return experiment, results


def run_first_price_experiment():
    """Run an experiment with first-price auction."""
    print("\n=== First-Price Auction Experiment ===\n")
    
    # Create experiment: agent 0 learns in first-price auction
    config = AuctionExperimentConfig.create_simple_experiment(
        experiment_name="first_price_auction",
        num_agents=3,
        trainable_agents=[0],
        agent_type="linear",
        valuation_type="deterministic",
        utility_type="linear",
        algorithm="SAC",  # Try different algorithm
        max_steps=50,
        auction_type="first_price",
        num_iterations=50,
        checkpoint_freq=10,
        verbose=1,
        # Agent configuration
        lambda_param=0.9,
        value=0.5,
    )
    
    experiment = AuctionExperiment(config)
    
    # Run training
    results = experiment.run()
    
    # Save experiment
    experiment.save_experiment("results/first_price_experiment.json")
    
    return experiment, results


def run_valuation_experiment():
    """Run an experiment with different valuation types."""
    print("\n=== Different Valuation Types Experiment ===\n")
    
    # For mixed agent types, we'll need to use create_mixed_experiment
    # or create individual agent configs manually
    from auction_gym.experiments.experiment import AgentConfig
    from gymnasium import spaces
    
    # Create individual agent configs for different valuation types
    action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    
    agent_configs = [
        # Agent 0: trainable with deterministic valuation
        AgentConfig(
            agent_id=0,
            agent_type="linear",
            action_space=action_space,
            valuation_type="deterministic",
            utility_type="linear",
            is_trainable=True,
            agent_params={"lambda_param": 1.0},
            valuation_params={"value": 0.5},
            utility_params={}
        ),
        # Agent 1: fixed with uniform valuation
        AgentConfig(
            agent_id=1,
            agent_type="linear",
            action_space=action_space,
            valuation_type="uniform",
            utility_type="linear",
            is_trainable=False,
            agent_params={"lambda_param": 0.8},
            valuation_params={"low": 0.0, "high": 1.0},
            utility_params={}
        ),
        # Agent 2: fixed with normal valuation
        AgentConfig(
            agent_id=2,
            agent_type="linear",
            action_space=action_space,
            valuation_type="normal",
            utility_type="linear",
            is_trainable=False,
            agent_params={"lambda_param": 0.9},
            valuation_params={"mean": 0.6, "std": 0.1},
            utility_params={}
        ),
    ]
    
    config = AuctionExperimentConfig.create_from_agent_configs(
        experiment_name="different_valuations",
        agent_configs=agent_configs,
        algorithm="PPO",
        max_steps=50,
        auction_type="second_price",
        num_iterations=50,
        checkpoint_freq=10,
        verbose=1,
    )
    
    experiment = AuctionExperiment(config)
    
    # Run training
    results = experiment.run()
    
    # Save experiment
    experiment.save_experiment("results/valuation_experiment.json")
    
    return experiment, results


def main():
    """Run all experiments."""
    print("Starting Auction Gym Experiments\n")
    
    # Create results directory
    import os
    os.makedirs("results", exist_ok=True)
    
    # Run experiments
    experiments = []
    
    try:
        # Single agent experiment
        exp1, results1 = run_single_agent_experiment()
        experiments.append(("Single Agent", exp1, results1))
        
        # Multi-agent experiment
        exp2, results2 = run_multi_agent_experiment()
        experiments.append(("Multi-Agent", exp2, results2))
        
        # First-price auction experiment
        exp3, results3 = run_first_price_experiment()
        experiments.append(("First-Price", exp3, results3))
        
        # Different valuations experiment
        exp4, results4 = run_valuation_experiment()
        experiments.append(("Different Valuations", exp4, results4))
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    
    for name, exp, results in experiments:
        print(f"\n{name} Experiment:")
        print(f"  Algorithm: {exp.algorithm}")
        print(f"  Learning agents: {exp.learning_agent_ids}")
        print(f"  Auction type: {exp.auction_type}")
        print(f"  Results: {results.checkpoint_dir if hasattr(results, 'checkpoint_dir') else 'Training completed'}")
    
    print(f"\nAll experiments completed! Results saved in 'results/' directory.")
    
    # Clean up Ray
    import ray
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main() 
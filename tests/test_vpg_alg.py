import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from gymnasium import spaces

from auction_gym.algorithms.vpg import VPG, VPGConfig
from auction_gym.modules.linear_bidder import LBTorchRLModule
from auction_gym.learners.vpg_learner import VPGTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig


class TestVPGConfig:
    """Test VPGConfig initialization and configuration methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = VPGConfig()
    
    def test_default_initialization(self):
        """Test that VPGConfig initializes with correct default values."""
        assert self.config.report_mean_weights is True
        assert self.config.num_episodes_per_train_batch == 10
        assert self.config.batch_mode == "complete_episodes"
        assert self.config.num_env_runners == 1
    
    def test_training_configuration(self):
        """Test training configuration method."""
        config = self.config.training(num_episodes_per_train_batch=20)
        
        assert config.num_episodes_per_train_batch == 20
        assert config is self.config  # Should return self
    
    def test_get_default_rl_module_spec_torch(self):
        """Test RL module spec generation for torch framework."""
        self.config.framework_str = "torch"
        spec = self.config.get_default_rl_module_spec()
        
        assert spec.module_class == LBTorchRLModule
        assert spec.model_config == {}
    
    def test_get_default_rl_module_spec_unsupported_framework(self):
        """Test RL module spec generation for unsupported framework."""
        self.config.framework_str = "tensorflow"
        
        with pytest.raises(ValueError, match="Unsupported framework: tensorflow"):
            self.config.get_default_rl_module_spec()
    
    def test_get_default_learner_class_torch(self):
        """Test learner class selection for torch framework."""
        self.config.framework_str = "torch"
        learner_class = self.config.get_default_learner_class()
        
        assert learner_class == VPGTorchLearner
    
    def test_get_default_learner_class_unsupported_framework(self):
        """Test learner class selection for unsupported framework."""
        self.config.framework_str = "tensorflow"
        
        with pytest.raises(ValueError, match="Unsupported framework: tensorflow"):
            self.config.get_default_learner_class()


class TestVPGAlgorithm:
    """Test VPG algorithm class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = VPGConfig()
        # Don't create the algorithm instance here - only create it when needed
    
    def test_get_default_config(self):
        """Test that get_default_config returns VPGConfig instance."""
        config = VPG.get_default_config()
        
        assert isinstance(config, VPGConfig)
        assert config.report_mean_weights is True
        assert config.num_episodes_per_train_batch == 10
    
    @patch('auction_gym.algorithms.vpg.VPG.__init__', return_value=None)
    @patch('auction_gym.algorithms.vpg.VPG._sample_episodes')
    @patch('auction_gym.algorithms.vpg.VPG.learner_group')
    @patch('auction_gym.algorithms.vpg.VPG.env_runner_group')
    @patch('auction_gym.algorithms.vpg.VPG.metrics')
    def test_training_step_basic_flow(self, mock_metrics, mock_env_runner_group, 
                                    mock_learner_group, mock_sample_episodes, mock_init):
        """Test basic training step flow."""
        # Create algorithm instance with mocked initialization
        self.algorithm = VPG(self.config)
        
        # Mock sample episodes - create episodes that have len()
        mock_episode1 = Mock()
        mock_episode1.__len__ = Mock(return_value=10)
        mock_episode2 = Mock()
        mock_episode2.__len__ = Mock(return_value=15)
        mock_episodes = [mock_episode1, mock_episode2]
        mock_env_runner_results = [{'test': 'result'}]
        mock_sample_episodes.return_value = (mock_episodes, mock_env_runner_results)
        
        # Mock learner group update
        mock_learner_results = {'loss': 0.5}
        mock_learner_group.update.return_value = mock_learner_results
        
        # Mock metrics
        mock_metrics.peek.return_value = 100
        mock_metrics.log_time.return_value.__enter__ = Mock()
        mock_metrics.log_time.return_value.__exit__ = Mock()
        mock_metrics.log_value.return_value = None  # Mock log_value method
        
        # Mock env runner group sync
        mock_env_runner_group.sync_weights.return_value = None
        
        # Call training step
        self.algorithm.config = self.config
        self.algorithm.training_step()
        
        # Verify calls
        mock_sample_episodes.assert_called_once()
        mock_learner_group.update.assert_called_once()
        mock_env_runner_group.sync_weights.assert_called_once()
        
        # Verify that log_value was called with the correct sum of episode lengths
        expected_sum = 10 + 15  # sum of the two episode lengths
        mock_metrics.log_value.assert_any_call(
            "episode_timesteps_sampled_mean_win100",
            expected_sum,
            reduce="mean",
            window=100,
        )
        mock_metrics.log_value.assert_any_call(
            "episode_timesteps_sampled_ema",
            expected_sum,
            ema_coeff=0.1,
        )
    
    @patch('auction_gym.algorithms.vpg.VPG.__init__', return_value=None)
    def test_sample_episodes_single_env_runner(self, mock_init):
        """Test episode sampling with single environment runner."""
        # Create algorithm instance with mocked initialization
        self.algorithm = VPG(self.config)
        
        self.algorithm.config = self.config
        self.algorithm.config.num_env_runners = 1
        self.algorithm.config.num_episodes_per_train_batch = 10
        
        # Mock env runner group
        mock_env_runner = Mock()
        mock_env_runner.sample.return_value = [Mock() for _ in range(10)]
        mock_env_runner.get_metrics.return_value = {'test': 'metrics'}
        
        self.algorithm.env_runner_group = Mock()
        self.algorithm.env_runner_group.num_remote_workers.return_value = 0
        self.algorithm.env_runner_group.foreach_env_runner.return_value = [
            ([Mock() for _ in range(10)], {'test': 'metrics'})
        ]
        
        episodes, stats = self.algorithm._sample_episodes()
        
        assert len(episodes) == 10
        assert len(stats) == 1
    
    @patch('auction_gym.algorithms.vpg.VPG.__init__', return_value=None)
    def test_sample_episodes_multiple_env_runners(self, mock_init):
        """Test episode sampling with multiple environment runners."""
        # Create algorithm instance with mocked initialization
        self.algorithm = VPG(self.config)
        
        self.algorithm.config = self.config
        self.algorithm.config.num_env_runners = 2
        self.algorithm.config.num_episodes_per_train_batch = 10
        
        # Mock env runner group
        self.algorithm.env_runner_group = Mock()
        self.algorithm.env_runner_group.num_remote_workers.return_value = 1
        self.algorithm.env_runner_group.foreach_env_runner.return_value = [
            ([Mock() for _ in range(5)], {'test': 'metrics1'}),
            ([Mock() for _ in range(5)], {'test': 'metrics2'})
        ]
        
        episodes, stats = self.algorithm._sample_episodes()
        
        assert len(episodes) == 10
        assert len(stats) == 2


class TestVPGAlgorithmIntegration:
    """Integration tests for VPG algorithm with real components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    def test_config_with_rl_module_spec(self):
        """Test that config works with RL module spec."""
        config = VPGConfig()
        config.framework_str = "torch"
        
        spec = config.get_default_rl_module_spec()
        
        # Test that we can create an RL module from the spec
        rl_module = spec.module_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config=spec.model_config
        )
        
        assert isinstance(rl_module, LBTorchRLModule)
        assert rl_module.observation_space == self.observation_space
        assert rl_module.action_space == self.action_space
    
    def test_config_with_learner_class(self):
        """Test that config works with learner class."""
        config = VPGConfig()
        config.framework_str = "torch"
        
        learner_class = config.get_default_learner_class()
        
        # Test that we can instantiate the learner class
        # Note: This is a simplified test since full learner instantiation
        # requires complex RLlib setup
        assert learner_class == VPGTorchLearner
        assert issubclass(learner_class, VPGTorchLearner.__bases__[0])


class TestVPGAlgorithmPerformance:
    """Performance tests for VPG algorithm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = VPGConfig()
        # Don't create algorithm instance here - create it in each test with proper mocking
    
    @patch('auction_gym.algorithms.vpg.VPG.__init__', return_value=None)
    @pytest.mark.parametrize("num_episodes", [1, 10, 100])
    def test_sample_episodes_scaling(self, mock_init, num_episodes):
        """Test that episode sampling scales with different batch sizes."""
        # Create algorithm instance with mocked initialization
        self.algorithm = VPG(self.config)
        
        self.algorithm.config = self.config
        self.algorithm.config.num_episodes_per_train_batch = num_episodes
        self.algorithm.config.num_env_runners = 1
        
        # Mock env runner group
        self.algorithm.env_runner_group = Mock()
        self.algorithm.env_runner_group.num_remote_workers.return_value = 0
        self.algorithm.env_runner_group.foreach_env_runner.return_value = [
            ([Mock() for _ in range(num_episodes)], {'test': 'metrics'})
        ]
        
        episodes, stats = self.algorithm._sample_episodes()
        
        assert len(episodes) == num_episodes
        assert len(stats) == 1
    
    @patch('auction_gym.algorithms.vpg.VPG.__init__', return_value=None)
    @pytest.mark.parametrize("num_env_runners", [1, 2, 4])
    def test_multiple_env_runners_scaling(self, mock_init, num_env_runners):
        """Test that algorithm scales with multiple environment runners."""
        # Create algorithm instance with mocked initialization
        self.algorithm = VPG(self.config)
        
        self.algorithm.config = self.config
        self.algorithm.config.num_env_runners = num_env_runners
        self.algorithm.config.num_episodes_per_train_batch = 10
        
        # Calculate how many episodes each env runner should return
        episodes_per_runner = 10 // num_env_runners
        # Calculate the actual total episodes (accounting for integer division)
        expected_total_episodes = episodes_per_runner * num_env_runners
        
        # Mock env runner group
        self.algorithm.env_runner_group = Mock()
        self.algorithm.env_runner_group.num_remote_workers.return_value = num_env_runners - 1
        self.algorithm.env_runner_group.foreach_env_runner.return_value = [
            ([Mock() for _ in range(episodes_per_runner)], {'test': f'metrics{i}'})
            for i in range(num_env_runners)
        ]
        
        episodes, stats = self.algorithm._sample_episodes()
        
        assert len(episodes) == expected_total_episodes
        assert len(stats) == num_env_runners 
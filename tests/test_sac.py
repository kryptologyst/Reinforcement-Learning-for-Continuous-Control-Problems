"""Unit tests for SAC algorithm."""

import pytest
import torch
import numpy as np
import gymnasium as gym

from src.algorithms.sac import SACAgent, GaussianPolicy, QNetwork
from src.utils.utils import set_seed, get_device


class TestGaussianPolicy:
    """Test cases for GaussianPolicy network."""
    
    def test_policy_initialization(self):
        """Test policy network initialization."""
        state_dim = 3
        action_dim = 2
        policy = GaussianPolicy(state_dim, action_dim)
        
        assert policy.state_dim == state_dim
        assert policy.action_dim == action_dim
        assert len(policy.network) > 0
    
    def test_policy_forward(self):
        """Test policy forward pass."""
        state_dim = 3
        action_dim = 2
        policy = GaussianPolicy(state_dim, action_dim)
        
        # Test with single state
        state = torch.randn(1, state_dim)
        mean, log_std = policy(state)
        
        assert mean.shape == (1, action_dim)
        assert log_std.shape == (1, action_dim)
        assert torch.all(log_std >= policy.log_std_min)
        assert torch.all(log_std <= policy.log_std_max)
    
    def test_policy_sample(self):
        """Test policy sampling."""
        state_dim = 3
        action_dim = 2
        policy = GaussianPolicy(state_dim, action_dim)
        
        state = torch.randn(1, state_dim)
        action, log_prob, mean = policy.sample(state)
        
        assert action.shape == (1, action_dim)
        assert log_prob.shape == (1, 1)
        assert mean.shape == (1, action_dim)
        assert torch.all(action >= -1) and torch.all(action <= 1)  # tanh output
    
    def test_policy_batch_forward(self):
        """Test policy forward pass with batch."""
        state_dim = 3
        action_dim = 2
        policy = GaussianPolicy(state_dim, action_dim)
        
        batch_size = 5
        state = torch.randn(batch_size, state_dim)
        mean, log_std = policy(state)
        
        assert mean.shape == (batch_size, action_dim)
        assert log_std.shape == (batch_size, action_dim)


class TestQNetwork:
    """Test cases for QNetwork."""
    
    def test_q_network_initialization(self):
        """Test Q-network initialization."""
        state_dim = 3
        action_dim = 2
        q_network = QNetwork(state_dim, action_dim)
        
        assert q_network.state_dim == state_dim
        assert q_network.action_dim == action_dim
        assert len(q_network.network) > 0
    
    def test_q_network_forward(self):
        """Test Q-network forward pass."""
        state_dim = 3
        action_dim = 2
        q_network = QNetwork(state_dim, action_dim)
        
        state = torch.randn(1, state_dim)
        action = torch.randn(1, action_dim)
        q_value = q_network(state, action)
        
        assert q_value.shape == (1, 1)
    
    def test_q_network_batch_forward(self):
        """Test Q-network forward pass with batch."""
        state_dim = 3
        action_dim = 2
        q_network = QNetwork(state_dim, action_dim)
        
        batch_size = 5
        state = torch.randn(batch_size, state_dim)
        action = torch.randn(batch_size, action_dim)
        q_value = q_network(state, action)
        
        assert q_value.shape == (batch_size, 1)


class TestSACAgent:
    """Test cases for SACAgent."""
    
    @pytest.fixture
    def env(self):
        """Create test environment."""
        return gym.make("Pendulum-v1")
    
    @pytest.fixture
    def agent(self, env):
        """Create test agent."""
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        return SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=env.action_space,
            device=torch.device("cpu")
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.state_dim > 0
        assert agent.action_dim > 0
        assert agent.device is not None
        assert agent.buffer is not None
        assert agent.policy is not None
        assert agent.q1 is not None
        assert agent.q2 is not None
        assert agent.target_q1 is not None
        assert agent.target_q2 is not None
    
    def test_agent_select_action(self, agent, env):
        """Test action selection."""
        state = env.observation_space.sample()
        action = agent.select_action(state)
        
        assert len(action) == agent.action_dim
        assert isinstance(action, np.ndarray)
    
    def test_agent_select_action_deterministic(self, agent, env):
        """Test deterministic action selection."""
        state = env.observation_space.sample()
        action = agent.select_action(state, deterministic=True)
        
        assert len(action) == agent.action_dim
        assert isinstance(action, np.ndarray)
    
    def test_agent_add_experience(self, agent, env):
        """Test adding experience."""
        state = env.observation_space.sample()
        action = env.action_space.sample()
        reward = 1.0
        next_state = env.observation_space.sample()
        done = False
        
        initial_size = agent.buffer.size
        agent.add_experience(state, action, reward, next_state, done)
        
        assert agent.buffer.size == initial_size + 1
    
    def test_agent_update_empty_buffer(self, agent):
        """Test update with empty buffer."""
        losses = agent.update()
        
        assert losses["policy_loss"] == 0.0
        assert losses["q1_loss"] == 0.0
        assert losses["q2_loss"] == 0.0
        assert losses["alpha_loss"] == 0.0
    
    def test_agent_update_with_data(self, agent, env):
        """Test update with data in buffer."""
        # Add enough experiences to buffer
        for _ in range(300):  # More than batch_size
            state = env.observation_space.sample()
            action = env.action_space.sample()
            reward = np.random.randn()
            next_state = env.observation_space.sample()
            done = np.random.choice([True, False])
            
            agent.add_experience(state, action, reward, next_state, done)
        
        # Update agent
        losses = agent.update()
        
        assert "policy_loss" in losses
        assert "q1_loss" in losses
        assert "q2_loss" in losses
        assert "alpha_loss" in losses
        assert isinstance(losses["policy_loss"], float)
        assert isinstance(losses["q1_loss"], float)
        assert isinstance(losses["q2_loss"], float)
        assert isinstance(losses["alpha_loss"], float)
    
    def test_agent_save_load(self, agent, tmp_path):
        """Test agent save and load."""
        # Add some training data
        agent.policy_losses = [1.0, 2.0, 3.0]
        agent.q1_losses = [0.5, 1.5, 2.5]
        agent.q2_losses = [0.3, 1.3, 2.3]
        agent.alpha_losses = [0.1, 0.2, 0.3]
        
        # Save agent
        save_path = tmp_path / "test_sac_agent.pth"
        agent.save(str(save_path))
        
        # Create new agent and load
        new_agent = SACAgent(
            state_dim=agent.state_dim,
            action_dim=agent.action_dim,
            action_space=agent.action_space,
            device=agent.device
        )
        
        new_agent.load(str(save_path))
        
        # Check that losses were loaded
        assert new_agent.policy_losses == agent.policy_losses
        assert new_agent.q1_losses == agent.q1_losses
        assert new_agent.q2_losses == agent.q2_losses
        assert new_agent.alpha_losses == agent.alpha_losses
        assert new_agent.alpha == agent.alpha
    
    def test_agent_auto_entropy_tuning(self, env):
        """Test automatic entropy tuning."""
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=env.action_space,
            device=torch.device("cpu"),
            auto_entropy=True
        )
        
        assert agent.auto_entropy is True
        assert hasattr(agent, 'log_alpha')
        assert hasattr(agent, 'alpha_optimizer')
        assert agent.target_entropy < 0  # Should be negative for continuous actions
    
    def test_agent_manual_entropy(self, env):
        """Test manual entropy coefficient."""
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=env.action_space,
            device=torch.device("cpu"),
            auto_entropy=False,
            alpha=0.5
        )
        
        assert agent.auto_entropy is False
        assert agent.alpha == 0.5
        assert not hasattr(agent, 'log_alpha')
        assert not hasattr(agent, 'alpha_optimizer')


if __name__ == "__main__":
    pytest.main([__file__])

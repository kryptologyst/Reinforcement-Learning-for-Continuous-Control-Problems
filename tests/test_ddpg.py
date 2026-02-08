"""Unit tests for DDPG algorithm."""

import pytest
import torch
import numpy as np
import gymnasium as gym

from src.algorithms.ddpg import DDPGAgent, Actor, Critic, ReplayBuffer, OUNoise
from src.utils.utils import set_seed, get_device


class TestActor:
    """Test cases for Actor network."""
    
    def test_actor_initialization(self):
        """Test actor network initialization."""
        state_dim = 3
        action_dim = 2
        actor = Actor(state_dim, action_dim)
        
        assert actor.state_dim == state_dim
        assert actor.action_dim == action_dim
        assert len(actor.network) > 0
    
    def test_actor_forward(self):
        """Test actor forward pass."""
        state_dim = 3
        action_dim = 2
        actor = Actor(state_dim, action_dim)
        
        # Test with single state
        state = torch.randn(1, state_dim)
        action = actor(state)
        
        assert action.shape == (1, action_dim)
        assert torch.all(action >= -1) and torch.all(action <= 1)  # tanh output
    
    def test_actor_batch_forward(self):
        """Test actor forward pass with batch."""
        state_dim = 3
        action_dim = 2
        actor = Actor(state_dim, action_dim)
        
        # Test with batch
        batch_size = 5
        state = torch.randn(batch_size, state_dim)
        action = actor(state)
        
        assert action.shape == (batch_size, action_dim)
        assert torch.all(action >= -1) and torch.all(action <= 1)


class TestCritic:
    """Test cases for Critic network."""
    
    def test_critic_initialization(self):
        """Test critic network initialization."""
        state_dim = 3
        action_dim = 2
        critic = Critic(state_dim, action_dim)
        
        assert critic.state_dim == state_dim
        assert critic.action_dim == action_dim
        assert len(critic.network) > 0
    
    def test_critic_forward(self):
        """Test critic forward pass."""
        state_dim = 3
        action_dim = 2
        critic = Critic(state_dim, action_dim)
        
        state = torch.randn(1, state_dim)
        action = torch.randn(1, action_dim)
        q_value = critic(state, action)
        
        assert q_value.shape == (1, 1)
    
    def test_critic_batch_forward(self):
        """Test critic forward pass with batch."""
        state_dim = 3
        action_dim = 2
        critic = Critic(state_dim, action_dim)
        
        batch_size = 5
        state = torch.randn(batch_size, state_dim)
        action = torch.randn(batch_size, action_dim)
        q_value = critic(state, action)
        
        assert q_value.shape == (batch_size, 1)


class TestReplayBuffer:
    """Test cases for ReplayBuffer."""
    
    def test_buffer_initialization(self):
        """Test replay buffer initialization."""
        capacity = 1000
        state_dim = 3
        action_dim = 2
        device = torch.device("cpu")
        
        buffer = ReplayBuffer(capacity, state_dim, action_dim, device)
        
        assert buffer.capacity == capacity
        assert buffer.size == 0
        assert buffer.ptr == 0
    
    def test_buffer_add(self):
        """Test adding experiences to buffer."""
        capacity = 1000
        state_dim = 3
        action_dim = 2
        device = torch.device("cpu")
        
        buffer = ReplayBuffer(capacity, state_dim, action_dim, device)
        
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = 1.0
        next_state = np.random.randn(state_dim)
        done = False
        
        buffer.add(state, action, reward, next_state, done)
        
        assert buffer.size == 1
        assert buffer.ptr == 1
    
    def test_buffer_sample(self):
        """Test sampling from buffer."""
        capacity = 1000
        state_dim = 3
        action_dim = 2
        device = torch.device("cpu")
        
        buffer = ReplayBuffer(capacity, state_dim, action_dim, device)
        
        # Add some experiences
        for _ in range(10):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            done = np.random.choice([True, False])
            
            buffer.add(state, action, reward, next_state, done)
        
        # Sample batch
        batch_size = 5
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        assert states.shape == (batch_size, state_dim)
        assert actions.shape == (batch_size, action_dim)
        assert rewards.shape == (batch_size, 1)
        assert next_states.shape == (batch_size, state_dim)
        assert dones.shape == (batch_size, 1)


class TestOUNoise:
    """Test cases for OUNoise."""
    
    def test_noise_initialization(self):
        """Test OU noise initialization."""
        action_dim = 2
        noise = OUNoise(action_dim)
        
        assert noise.action_dim == action_dim
        assert len(noise.state) == action_dim
    
    def test_noise_sample(self):
        """Test noise sampling."""
        action_dim = 2
        noise = OUNoise(action_dim)
        
        sample = noise.sample()
        
        assert len(sample) == action_dim
        assert isinstance(sample, np.ndarray)
    
    def test_noise_reset(self):
        """Test noise reset."""
        action_dim = 2
        noise = OUNoise(action_dim)
        
        # Sample to change state
        noise.sample()
        
        # Reset
        noise.reset()
        
        assert np.allclose(noise.state, noise.mu)


class TestDDPGAgent:
    """Test cases for DDPGAgent."""
    
    @pytest.fixture
    def env(self):
        """Create test environment."""
        return gym.make("Pendulum-v1")
    
    @pytest.fixture
    def agent(self, env):
        """Create test agent."""
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        return DDPGAgent(
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
        assert agent.noise is not None
    
    def test_agent_select_action(self, agent, env):
        """Test action selection."""
        state = env.observation_space.sample()
        action = agent.select_action(state)
        
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
        
        assert losses["actor_loss"] == 0.0
        assert losses["critic_loss"] == 0.0
    
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
        
        assert "actor_loss" in losses
        assert "critic_loss" in losses
        assert isinstance(losses["actor_loss"], float)
        assert isinstance(losses["critic_loss"], float)
    
    def test_agent_save_load(self, agent, tmp_path):
        """Test agent save and load."""
        # Add some training data
        agent.actor_losses = [1.0, 2.0, 3.0]
        agent.critic_losses = [0.5, 1.5, 2.5]
        
        # Save agent
        save_path = tmp_path / "test_agent.pth"
        agent.save(str(save_path))
        
        # Create new agent and load
        new_agent = DDPGAgent(
            state_dim=agent.state_dim,
            action_dim=agent.action_dim,
            action_space=agent.action_space,
            device=agent.device
        )
        
        new_agent.load(str(save_path))
        
        # Check that losses were loaded
        assert new_agent.actor_losses == agent.actor_losses
        assert new_agent.critic_losses == agent.critic_losses


if __name__ == "__main__":
    pytest.main([__file__])

"""Deep Deterministic Policy Gradient (DDPG) implementation for continuous control."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

from ..utils.utils import get_device, clip_actions

logger = logging.getLogger(__name__)


class Actor(nn.Module):
    """Actor network for DDPG."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        output_activation: str = "tanh"
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        if output_activation != "none":
            layers.append(self._get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "none": nn.Identity()
        }
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network."""
        return self.network(state)


class Critic(nn.Module):
    """Critic network for DDPG."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu"
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic network."""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DDPG."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate tensors
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """Add experience to buffer."""
        self.states[self.ptr] = torch.from_numpy(state).float()
        self.actions[self.ptr] = torch.from_numpy(action).float()
        self.rewards[self.ptr] = torch.tensor(reward, dtype=torch.float32)
        self.next_states[self.ptr] = torch.from_numpy(next_state).float()
        self.dones[self.ptr] = torch.tensor(done, dtype=torch.bool)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch from buffer."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )


class OUNoise:
    """Ornstein-Uhlenbeck noise for exploration."""
    
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu
        self.reset()
    
    def reset(self) -> None:
        """Reset noise state."""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self) -> np.ndarray:
        """Sample noise."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class DDPGAgent:
    """DDPG agent for continuous control."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_space: Optional[object] = None,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = 1000000,
        hidden_dims: List[int] = [256, 256],
        device: Optional[torch.device] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device or get_device()
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_actor = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Initialize target networks
        self._hard_update(self.target_actor, self.actor)
        self._hard_update(self.target_critic, self.critic)
        
        # Experience replay buffer
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim, self.device)
        
        # Exploration noise
        self.noise = OUNoise(action_dim)
        
        # Training statistics
        self.actor_losses = []
        self.critic_losses = []
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using actor network."""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if add_noise:
            noise = self.noise.sample()
            action += noise
        
        # Clip action to valid range
        if self.action_space is not None:
            action = clip_actions(action, self.action_space)
        
        return action
    
    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                      next_state: np.ndarray, done: bool) -> None:
        """Add experience to replay buffer."""
        self.buffer.add(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """Update actor and critic networks."""
        if self.buffer.size < self.batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0}
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, next_actions)
            target_q_values = rewards + (1 - dones.float()) * self.gamma * target_q_values
        
        current_q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        # Store losses
        actor_loss_val = actor_loss.item()
        critic_loss_val = critic_loss.item()
        self.actor_losses.append(actor_loss_val)
        self.critic_losses.append(critic_loss_val)
        
        return {"actor_loss": actor_loss_val, "critic_loss": critic_loss_val}
    
    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)
    
    def _hard_update(self, target: nn.Module, source: nn.Module) -> None:
        """Hard update target network."""
        target.load_state_dict(source.state_dict())
    
    def save(self, filepath: str) -> None:
        """Save agent state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'noise_state': self.noise.state,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.noise.state = checkpoint['noise_state']
        self.actor_losses = checkpoint['actor_losses']
        self.critic_losses = checkpoint['critic_losses']

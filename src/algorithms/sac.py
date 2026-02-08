"""Soft Actor-Critic (SAC) implementation for continuous control."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

from ..utils.utils import get_device, clip_actions

logger = logging.getLogger(__name__)


class GaussianPolicy(nn.Module):
    """Gaussian policy network for SAC."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
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
        
        nn.init.xavier_uniform_(self.mean_layer.weight)
        nn.init.constant_(self.mean_layer.bias, 0)
        nn.init.xavier_uniform_(self.log_std_layer.weight)
        nn.init.constant_(self.log_std_layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the policy network."""
        x = self.network(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean


class QNetwork(nn.Module):
    """Q-network for SAC."""
    
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
        """Forward pass through the Q-network."""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class SACAgent:
    """Soft Actor-Critic agent for continuous control."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_space: Optional[object] = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy: bool = True,
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
        self.alpha = alpha
        self.auto_entropy = auto_entropy
        self.batch_size = batch_size
        self.device = device or get_device()
        
        # Networks
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Target networks
        self.target_q1 = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_q2 = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Initialize target networks
        self._hard_update(self.target_q1, self.q1)
        self._hard_update(self.target_q2, self.q2)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        # Automatic entropy tuning
        if self.auto_entropy:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # Experience replay buffer (reuse from DDPG)
        from .ddpg import ReplayBuffer
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim, self.device)
        
        # Training statistics
        self.policy_losses = []
        self.q1_losses = []
        self.q2_losses = []
        self.alpha_losses = []
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using policy network."""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            
            if not add_noise:
                _, _, action = self.policy.sample(state_tensor)
            else:
                action, _, _ = self.policy.sample(state_tensor)
            
            action = action.cpu().numpy()[0]
        
        # Clip action to valid range
        if self.action_space is not None:
            action = clip_actions(action, self.action_space)
        
        return action
    
    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                      next_state: np.ndarray, done: bool) -> None:
        """Add experience to replay buffer."""
        self.buffer.add(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """Update policy and Q-networks."""
        if self.buffer.size < self.batch_size:
            return {"policy_loss": 0.0, "q1_loss": 0.0, "q2_loss": 0.0, "alpha_loss": 0.0}
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Update Q-networks
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_states)
            target_q1 = self.target_q1(next_states, next_actions)
            target_q2 = self.target_q2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones.float()) * self.gamma * target_q
        
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy
        new_actions, log_probs, _ = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update alpha (entropy coefficient)
        alpha_loss = 0.0
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        self._soft_update(self.target_q1, self.q1)
        self._soft_update(self.target_q2, self.q2)
        
        # Store losses
        policy_loss_val = policy_loss.item()
        q1_loss_val = q1_loss.item()
        q2_loss_val = q2_loss.item()
        alpha_loss_val = alpha_loss.item() if self.auto_entropy else 0.0
        
        self.policy_losses.append(policy_loss_val)
        self.q1_losses.append(q1_loss_val)
        self.q2_losses.append(q2_loss_val)
        if self.auto_entropy:
            self.alpha_losses.append(alpha_loss_val)
        
        return {
            "policy_loss": policy_loss_val,
            "q1_loss": q1_loss_val,
            "q2_loss": q2_loss_val,
            "alpha_loss": alpha_loss_val
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)
    
    def _hard_update(self, target: nn.Module, source: nn.Module) -> None:
        """Hard update target network."""
        target.load_state_dict(source.state_dict())
    
    def save(self, filepath: str) -> None:
        """Save agent state."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'target_q1_state_dict': self.target_q1.state_dict(),
            'target_q2_state_dict': self.target_q2.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'policy_losses': self.policy_losses,
            'q1_losses': self.q1_losses,
            'q2_losses': self.q2_losses,
            'alpha': self.alpha
        }
        
        if self.auto_entropy:
            checkpoint.update({
                'log_alpha': self.log_alpha,
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
                'alpha_losses': self.alpha_losses
            })
        
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.target_q1.load_state_dict(checkpoint['target_q1_state_dict'])
        self.target_q2.load_state_dict(checkpoint['target_q2_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        self.policy_losses = checkpoint['policy_losses']
        self.q1_losses = checkpoint['q1_losses']
        self.q2_losses = checkpoint['q2_losses']
        self.alpha = checkpoint['alpha']
        
        if self.auto_entropy and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha_losses = checkpoint['alpha_losses']

"""Utility functions for the RL continuous control project."""

import random
import numpy as np
import torch
import gymnasium as gym
from typing import Any, Dict, Optional, Tuple, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def set_env_seed(env: gym.Env, seed: int) -> None:
    """Set seed for gymnasium environment.
    
    Args:
        env: Gymnasium environment
        seed: Random seed value
    """
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def normalize_rewards(rewards: np.ndarray, running_mean: float, running_var: float, 
                     eps: float = 1e-8) -> Tuple[np.ndarray, float, float]:
    """Normalize rewards using running statistics.
    
    Args:
        rewards: Array of rewards
        running_mean: Current running mean
        running_var: Current running variance
        eps: Small constant for numerical stability
        
    Returns:
        Tuple of (normalized_rewards, updated_mean, updated_var)
    """
    batch_mean = np.mean(rewards)
    batch_var = np.var(rewards)
    
    # Update running statistics
    updated_mean = 0.99 * running_mean + 0.01 * batch_mean
    updated_var = 0.99 * running_var + 0.01 * batch_var
    
    # Normalize
    normalized_rewards = (rewards - updated_mean) / np.sqrt(updated_var + eps)
    
    return normalized_rewards, updated_mean, updated_var


def clip_actions(actions: np.ndarray, action_space: gym.Space) -> np.ndarray:
    """Clip actions to valid action space bounds.
    
    Args:
        actions: Array of actions
        action_space: Gymnasium action space
        
    Returns:
        Clipped actions
    """
    if isinstance(action_space, gym.spaces.Box):
        return np.clip(actions, action_space.low, action_space.high)
    return actions


def compute_returns(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """Compute discounted returns.
    
    Args:
        rewards: Array of rewards
        gamma: Discount factor
        
    Returns:
        Array of discounted returns
    """
    returns = np.zeros_like(rewards)
    running_return = 0
    
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    return returns


def compute_gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray,
                gamma: float = 0.99, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation.
    
    Args:
        rewards: Array of rewards
        values: Array of value estimates
        next_values: Array of next state value estimates
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    advantages = np.zeros_like(rewards)
    running_advantage = 0
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        running_advantage = delta + gamma * lam * running_advantage
        advantages[t] = running_advantage
    
    returns = advantages + values
    return advantages, returns


class RunningStats:
    """Running statistics for normalization."""
    
    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps
        self.eps = eps
    
    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean += delta * batch_count / total_count
        self.var += (delta * delta * self.count * batch_count / total_count +
                    (batch_var - self.var) * batch_count / total_count)
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize data using running statistics."""
        return (x - self.mean) / np.sqrt(self.var + self.eps)
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize data using running statistics."""
        return x * np.sqrt(self.var + self.eps) + self.mean

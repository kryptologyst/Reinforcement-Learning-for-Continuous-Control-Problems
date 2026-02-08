#!/usr/bin/env python3
"""Simple training script for testing the RL implementation."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm

from src.algorithms.ddpg import DDPGAgent
from src.algorithms.sac import SACAgent
from src.utils.utils import set_seed, set_env_seed, get_device


def train_simple(env_name="Pendulum-v1", algorithm="ddpg", episodes=100):
    """Simple training function for testing."""
    print(f"Training {algorithm.upper()} on {env_name} for {episodes} episodes")
    
    # Set up
    device = get_device()
    set_seed(42)
    
    # Create environment
    env = gym.make(env_name)
    set_env_seed(env, 42)
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {device}")
    
    # Create agent
    agent_config = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_space": env.action_space,
        "device": device,
        "batch_size": 64,  # Smaller for testing
        "buffer_size": 10000  # Smaller for testing
    }
    
    if algorithm.lower() == "ddpg":
        agent = DDPGAgent(**agent_config)
    elif algorithm.lower() == "sac":
        agent = SACAgent(**agent_config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Training loop
    episode_rewards = []
    
    for episode in tqdm(range(episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state, add_noise=True)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Add experience
            agent.add_experience(state, action, reward, next_state, done)
            
            # Update agent
            losses = agent.update()
            
            episode_reward += reward
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode}: Average Reward (last 20): {avg_reward:.2f}")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    eval_rewards = []
    
    for _ in range(10):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, add_noise=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
        
        eval_rewards.append(episode_reward)
    
    print(f"Final Evaluation Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Training completed successfully!")
    
    env.close()
    return agent, episode_rewards


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple training test")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Environment name")
    parser.add_argument("--algorithm", type=str, default="ddpg", choices=["ddpg", "sac"], help="Algorithm")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    
    args = parser.parse_args()
    
    try:
        agent, rewards = train_simple(args.env, args.algorithm, args.episodes)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

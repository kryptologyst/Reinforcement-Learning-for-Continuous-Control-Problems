"""Evaluation script for continuous control RL agents."""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple
import logging

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.algorithms.ddpg import DDPGAgent
from src.algorithms.sac import SACAgent
from src.utils.utils import set_seed, set_env_seed, get_device

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator class for RL agents."""
    
    def __init__(
        self,
        env_name: str,
        algorithm: str,
        model_path: str,
        device: Optional[torch.device] = None,
        seed: int = 42
    ):
        self.env_name = env_name
        self.algorithm = algorithm
        self.model_path = model_path
        self.device = device or get_device()
        self.seed = seed
        
        # Set random seed
        set_seed(seed)
        
        # Create environment
        self.env = gym.make(env_name)
        set_env_seed(self.env, seed)
        
        # Get environment dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Create agent
        self.agent = self._create_agent()
        
        # Load model
        self._load_model()
    
    def _create_agent(self):
        """Create the appropriate agent based on algorithm."""
        agent_config = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_space": self.env.action_space,
            "device": self.device
        }
        
        if self.algorithm.lower() == "ddpg":
            return DDPGAgent(**agent_config)
        elif self.algorithm.lower() == "sac":
            return SACAgent(**agent_config)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _load_model(self) -> None:
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.agent.load(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")
    
    def evaluate(self, num_episodes: int = 100, render: bool = False) -> Dict:
        """Evaluate the agent."""
        logger.info(f"Evaluating agent for {num_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        episode_data = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            states = [state]
            actions = []
            rewards = []
            
            done = False
            while not done:
                action = self.agent.select_action(state, add_noise=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                
                if render and episode == 0:  # Render only first episode
                    self.env.render()
                    time.sleep(0.01)
                
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_data.append({
                "episode": episode,
                "reward": episode_reward,
                "length": episode_length,
                "states": np.array(states),
                "actions": np.array(actions),
                "rewards": np.array(rewards)
            })
        
        # Compute statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        
        # Compute confidence interval
        ci_95 = 1.96 * std_reward / np.sqrt(num_episodes)
        
        results = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "std_length": std_length,
            "ci_95": ci_95,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "episode_data": episode_data
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"95% CI: [{mean_reward - ci_95:.2f}, {mean_reward + ci_95:.2f}]")
        logger.info(f"Mean Length: {mean_length:.2f} ± {std_length:.2f}")
        
        return results
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None) -> None:
        """Plot evaluation results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Reward distribution
        axes[0, 0].hist(results["episode_rewards"], bins=20, alpha=0.7, color='blue')
        axes[0, 0].axvline(results["mean_reward"], color='red', linestyle='--', 
                          label=f'Mean: {results["mean_reward"]:.2f}')
        axes[0, 0].set_xlabel('Episode Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Episode Reward Distribution')
        axes[0, 0].legend()
        
        # Episode length distribution
        axes[0, 1].hist(results["episode_lengths"], bins=20, alpha=0.7, color='green')
        axes[0, 1].axvline(results["mean_length"], color='red', linestyle='--',
                          label=f'Mean: {results["mean_length"]:.2f}')
        axes[0, 1].set_xlabel('Episode Length')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Episode Length Distribution')
        axes[0, 1].legend()
        
        # Reward over episodes
        axes[1, 0].plot(results["episode_rewards"], alpha=0.7, color='blue')
        axes[1, 0].axhline(results["mean_reward"], color='red', linestyle='--',
                          label=f'Mean: {results["mean_reward"]:.2f}')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title('Episode Rewards Over Time')
        axes[1, 0].legend()
        
        # Length over episodes
        axes[1, 1].plot(results["episode_lengths"], alpha=0.7, color='green')
        axes[1, 1].axhline(results["mean_length"], color='red', linestyle='--',
                          label=f'Mean: {results["mean_length"]:.2f}')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Length')
        axes[1, 1].set_title('Episode Lengths Over Time')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def analyze_trajectory(self, episode_data: Dict, save_path: Optional[str] = None) -> None:
        """Analyze a single trajectory."""
        states = episode_data["states"]
        actions = episode_data["actions"]
        rewards = episode_data["rewards"]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # State trajectory
        for i in range(min(states.shape[1], 3)):  # Plot first 3 state dimensions
            axes[0, 0].plot(states[:, i], label=f'State {i}')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('State Value')
        axes[0, 0].set_title('State Trajectory')
        axes[0, 0].legend()
        
        # Action trajectory
        for i in range(min(actions.shape[1], 3)):  # Plot first 3 action dimensions
            axes[0, 1].plot(actions[:, i], label=f'Action {i}')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Action Value')
        axes[0, 1].set_title('Action Trajectory')
        axes[0, 1].legend()
        
        # Reward trajectory
        axes[1, 0].plot(rewards, color='red')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title('Reward Trajectory')
        
        # Cumulative reward
        cumulative_rewards = np.cumsum(rewards)
        axes[1, 1].plot(cumulative_rewards, color='purple')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].set_title('Cumulative Reward')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trajectory plot saved to {save_path}")
        
        plt.show()
    
    def close(self) -> None:
        """Close the environment."""
        self.env.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate RL agent for continuous control")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Environment name")
    parser.add_argument("--algorithm", type=str, default="ddpg", choices=["ddpg", "sac"], help="RL algorithm")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/mps/cpu/auto)")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    # Create evaluator
    evaluator = Evaluator(
        env_name=args.env,
        algorithm=args.algorithm,
        model_path=args.model,
        device=device,
        seed=args.seed
    )
    
    try:
        # Evaluate the agent
        results = evaluator.evaluate(num_episodes=args.episodes, render=args.render)
        
        # Plot results
        plot_path = f"assets/evaluation_results_{args.algorithm}_{args.env}.png" if args.save_plots else None
        evaluator.plot_results(results, save_path=plot_path)
        
        # Analyze first trajectory
        if results["episode_data"]:
            trajectory_path = f"assets/trajectory_{args.algorithm}_{args.env}.png" if args.save_plots else None
            evaluator.analyze_trajectory(results["episode_data"][0], save_path=trajectory_path)
        
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()

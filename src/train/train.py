"""Training script for continuous control RL agents."""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple
import logging

import gymnasium as gym
import numpy as np
import torch
import wandb
from tqdm import tqdm

from src.algorithms.ddpg import DDPGAgent
from src.algorithms.sac import SACAgent
from src.utils.utils import set_seed, set_env_seed, get_device, RunningStats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for RL agents."""
    
    def __init__(
        self,
        env_name: str,
        algorithm: str,
        config: Dict,
        device: Optional[torch.device] = None,
        use_wandb: bool = False
    ):
        self.env_name = env_name
        self.algorithm = algorithm
        self.config = config
        self.device = device or get_device()
        self.use_wandb = use_wandb
        
        # Set random seed
        set_seed(config.get("seed", 42))
        
        # Create environment
        self.env = gym.make(env_name)
        set_env_seed(self.env, config.get("seed", 42))
        
        # Get environment dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Create agent
        self.agent = self._create_agent()
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_lengths = []
        
        # Running statistics for normalization
        self.reward_stats = RunningStats((1,))
        self.state_stats = RunningStats(self.state_dim)
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project="rl-continuous-control",
                name=f"{algorithm}_{env_name}_{int(time.time())}",
                config=config
            )
    
    def _create_agent(self):
        """Create the appropriate agent based on algorithm."""
        agent_config = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_space": self.env.action_space,
            "device": self.device,
            **self.config.get("agent", {})
        }
        
        if self.algorithm.lower() == "ddpg":
            return DDPGAgent(**agent_config)
        elif self.algorithm.lower() == "sac":
            return SACAgent(**agent_config)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train(self, num_episodes: int, eval_freq: int = 100, save_freq: int = 500) -> Dict:
        """Train the agent."""
        logger.info(f"Starting training for {num_episodes} episodes")
        
        best_eval_reward = float('-inf')
        training_start_time = time.time()
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            episode_reward, episode_length = self._train_episode()
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Log training metrics
            if self.use_wandb:
                wandb.log({
                    "train/episode_reward": episode_reward,
                    "train/episode_length": episode_length,
                    "train/avg_reward_100": np.mean(self.episode_rewards[-100:]),
                    "train/avg_length_100": np.mean(self.episode_lengths[-100:]),
                    "episode": episode
                })
            
            # Evaluation
            if episode % eval_freq == 0:
                eval_reward, eval_length = self._evaluate(num_episodes=10)
                self.eval_rewards.append(eval_reward)
                self.eval_lengths.append(eval_length)
                
                logger.info(
                    f"Episode {episode}: Train Reward: {episode_reward:.2f}, "
                    f"Eval Reward: {eval_reward:.2f} ± {eval_length:.2f}"
                )
                
                if self.use_wandb:
                    wandb.log({
                        "eval/episode_reward": eval_reward,
                        "eval/episode_length": eval_length,
                        "eval/episode": episode
                    })
                
                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self._save_model("best_model.pth")
            
            # Periodic save
            if episode % save_freq == 0:
                self._save_model(f"checkpoint_episode_{episode}.pth")
        
        training_time = time.time() - training_start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Final evaluation
        final_eval_reward, final_eval_length = self._evaluate(num_episodes=50)
        
        return {
            "final_eval_reward": final_eval_reward,
            "final_eval_length": final_eval_length,
            "best_eval_reward": best_eval_reward,
            "training_time": training_time,
            "total_episodes": num_episodes
        }
    
    def _train_episode(self) -> Tuple[float, int]:
        """Train for one episode."""
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Update state statistics
        self.state_stats.update(state.reshape(1, -1))
        
        done = False
        while not done:
            # Select action
            action = self.agent.select_action(state, add_noise=True)
            
            # Take step
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Update reward statistics
            self.reward_stats.update(np.array([[reward]]))
            
            # Add experience to buffer
            self.agent.add_experience(state, action, reward, next_state, done)
            
            # Update agent
            losses = self.agent.update()
            
            # Log losses
            if self.use_wandb and losses:
                wandb.log(losses)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        return episode_reward, episode_length
    
    def _evaluate(self, num_episodes: int = 10) -> Tuple[float, float]:
        """Evaluate the agent."""
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            done = False
            while not done:
                action = self.agent.select_action(state, add_noise=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        return np.mean(eval_rewards), np.std(eval_rewards)
    
    def _save_model(self, filename: str) -> None:
        """Save the agent model."""
        os.makedirs("checkpoints", exist_ok=True)
        filepath = os.path.join("checkpoints", filename)
        self.agent.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def close(self) -> None:
        """Close the environment and finish logging."""
        self.env.close()
        if self.use_wandb:
            wandb.finish()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL agent for continuous control")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Environment name")
    parser.add_argument("--algorithm", type=str, default="ddpg", choices=["ddpg", "sac"], help="RL algorithm")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/mps/cpu/auto)")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    # Default configuration
    config = {
        "seed": args.seed,
        "agent": {
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 256,
            "buffer_size": 1000000,
            "hidden_dims": [256, 256]
        }
    }
    
    # Load config file if provided
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config.update(yaml.safe_load(f))
    
    # Create trainer
    trainer = Trainer(
        env_name=args.env,
        algorithm=args.algorithm,
        config=config,
        device=device,
        use_wandb=args.wandb
    )
    
    try:
        # Train the agent
        results = trainer.train(num_episodes=args.episodes)
        
        # Print results
        logger.info("Training Results:")
        logger.info(f"Final Evaluation Reward: {results['final_eval_reward']:.2f}")
        logger.info(f"Best Evaluation Reward: {results['best_eval_reward']:.2f}")
        logger.info(f"Training Time: {results['training_time']:.2f} seconds")
        
    finally:
        trainer.close()


if __name__ == "__main__":
    main()

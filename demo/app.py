"""Streamlit demo for continuous control RL agents."""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import torch
import gymnasium as gym
import os
from typing import Dict, List, Optional, Tuple

from src.algorithms.ddpg import DDPGAgent
from src.algorithms.sac import SACAgent
from src.utils.utils import set_seed, set_env_seed, get_device

# Set page config
st.set_page_config(
    page_title="RL Continuous Control Demo",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 Reinforcement Learning for Continuous Control</h1>', unsafe_allow_html=True)

# Safety disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Safety Notice</h4>
    <p><strong>This is a research/educational demonstration only.</strong> 
    Do not use these algorithms for production control of real-world systems without 
    extensive safety validation and risk assessment.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Environment selection
env_name = st.sidebar.selectbox(
    "Environment",
    ["Pendulum-v1", "MountainCarContinuous-v0", "BipedalWalker-v3"],
    help="Select the continuous control environment"
)

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["ddpg", "sac"],
    help="Select the RL algorithm"
)

# Model selection
model_files = []
if os.path.exists("checkpoints"):
    model_files = [f for f in os.listdir("checkpoints") if f.endswith(".pth")]
    model_files = ["None"] + model_files

model_file = st.sidebar.selectbox(
    "Trained Model",
    model_files,
    help="Select a trained model to load"
)

# Evaluation parameters
num_episodes = st.sidebar.slider(
    "Number of Evaluation Episodes",
    min_value=1,
    max_value=100,
    value=10,
    help="Number of episodes to run for evaluation"
)

render_episodes = st.sidebar.slider(
    "Episodes to Render",
    min_value=0,
    max_value=5,
    value=1,
    help="Number of episodes to render (0 = no rendering)"
)

seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=10000,
    value=42,
    help="Random seed for reproducibility"
)

# Main content
if model_file == "None":
    st.warning("Please select a trained model to evaluate.")
    st.stop()

# Load and evaluate agent
@st.cache_data
def load_and_evaluate(env_name: str, algorithm: str, model_path: str, 
                     num_episodes: int, seed: int) -> Dict:
    """Load agent and run evaluation."""
    try:
        # Set device
        device = get_device()
        
        # Set seed
        set_seed(seed)
        
        # Create environment
        env = gym.make(env_name)
        set_env_seed(env, seed)
        
        # Get environment dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Create agent
        agent_config = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_space": env.action_space,
            "device": device
        }
        
        if algorithm.lower() == "ddpg":
            agent = DDPGAgent(**agent_config)
        elif algorithm.lower() == "sac":
            agent = SACAgent(**agent_config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Load model
        agent.load(model_path)
        
        # Run evaluation
        episode_rewards = []
        episode_lengths = []
        episode_data = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            states = [state]
            actions = []
            rewards = []
            
            done = False
            while not done:
                action = agent.select_action(state, add_noise=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                
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
        
        env.close()
        
        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "episode_data": episode_data,
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths)
        }
    
    except Exception as e:
        st.error(f"Error during evaluation: {str(e)}")
        return None

# Run evaluation
if st.sidebar.button("Run Evaluation"):
    with st.spinner("Running evaluation..."):
        model_path = os.path.join("checkpoints", model_file)
        results = load_and_evaluate(env_name, algorithm, model_path, num_episodes, seed)
    
    if results is not None:
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Mean Reward",
                f"{results['mean_reward']:.2f}",
                delta=f"±{results['std_reward']:.2f}"
            )
        
        with col2:
            st.metric(
                "Mean Length",
                f"{results['mean_length']:.1f}",
                delta=f"±{results['std_length']:.1f}"
            )
        
        with col3:
            st.metric(
                "Best Reward",
                f"{max(results['episode_rewards']):.2f}"
            )
        
        with col4:
            st.metric(
                "Worst Reward",
                f"{min(results['episode_rewards']):.2f}"
            )
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "📈 Trajectories", "🎯 Action Analysis", "📋 Episode Details"])
        
        with tab1:
            # Performance plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Episode Rewards", "Episode Lengths", 
                              "Reward Distribution", "Length Distribution"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Episode rewards over time
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(results['episode_rewards']))),
                    y=results['episode_rewards'],
                    mode='lines+markers',
                    name='Episode Rewards',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Episode lengths over time
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(results['episode_lengths']))),
                    y=results['episode_lengths'],
                    mode='lines+markers',
                    name='Episode Lengths',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
            
            # Reward distribution
            fig.add_trace(
                go.Histogram(
                    x=results['episode_rewards'],
                    name='Reward Distribution',
                    marker_color='blue',
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Length distribution
            fig.add_trace(
                go.Histogram(
                    x=results['episode_lengths'],
                    name='Length Distribution',
                    marker_color='green',
                    opacity=0.7
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Trajectory analysis
            if results['episode_data']:
                episode_idx = st.selectbox(
                    "Select Episode",
                    range(len(results['episode_data'])),
                    format_func=lambda x: f"Episode {x+1} (Reward: {results['episode_data'][x]['reward']:.2f})"
                )
                
                episode_data = results['episode_data'][episode_idx]
                states = episode_data['states']
                actions = episode_data['actions']
                rewards = episode_data['rewards']
                
                # State trajectory
                fig_states = go.Figure()
                for i in range(min(states.shape[1], 4)):  # Plot first 4 state dimensions
                    fig_states.add_trace(go.Scatter(
                        y=states[:, i],
                        mode='lines',
                        name=f'State {i}',
                        line=dict(width=2)
                    ))
                
                fig_states.update_layout(
                    title="State Trajectory",
                    xaxis_title="Time Step",
                    yaxis_title="State Value",
                    height=400
                )
                st.plotly_chart(fig_states, use_container_width=True)
                
                # Action trajectory
                fig_actions = go.Figure()
                for i in range(min(actions.shape[1], 4)):  # Plot first 4 action dimensions
                    fig_actions.add_trace(go.Scatter(
                        y=actions[:, i],
                        mode='lines',
                        name=f'Action {i}',
                        line=dict(width=2)
                    ))
                
                fig_actions.update_layout(
                    title="Action Trajectory",
                    xaxis_title="Time Step",
                    yaxis_title="Action Value",
                    height=400
                )
                st.plotly_chart(fig_actions, use_container_width=True)
        
        with tab3:
            # Action analysis
            if results['episode_data']:
                all_actions = np.concatenate([ep['actions'] for ep in results['episode_data']])
                
                # Action statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Action Statistics")
                    action_stats = pd.DataFrame({
                        'Action Dimension': [f'Action {i}' for i in range(all_actions.shape[1])],
                        'Mean': np.mean(all_actions, axis=0),
                        'Std': np.std(all_actions, axis=0),
                        'Min': np.min(all_actions, axis=0),
                        'Max': np.max(all_actions, axis=0)
                    })
                    st.dataframe(action_stats, use_container_width=True)
                
                with col2:
                    st.subheader("Action Distribution")
                    action_dim = st.selectbox(
                        "Action Dimension",
                        range(all_actions.shape[1]),
                        format_func=lambda x: f"Action {x}"
                    )
                    
                    fig_action_dist = go.Figure()
                    fig_action_dist.add_trace(go.Histogram(
                        x=all_actions[:, action_dim],
                        name=f'Action {action_dim}',
                        marker_color='purple',
                        opacity=0.7
                    ))
                    
                    fig_action_dist.update_layout(
                        title=f"Action {action_dim} Distribution",
                        xaxis_title="Action Value",
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig_action_dist, use_container_width=True)
        
        with tab4:
            # Episode details table
            episode_df = pd.DataFrame([
                {
                    'Episode': i+1,
                    'Reward': ep['reward'],
                    'Length': ep['length'],
                    'Mean Reward per Step': ep['reward'] / ep['length'] if ep['length'] > 0 else 0
                }
                for i, ep in enumerate(results['episode_data'])
            ])
            
            st.subheader("Episode Details")
            st.dataframe(episode_df, use_container_width=True)
            
            # Download results
            csv = episode_df.to_csv(index=False)
            st.download_button(
                label="Download Episode Data",
                data=csv,
                file_name=f"episode_results_{algorithm}_{env_name}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Reinforcement Learning for Continuous Control - Research/Educational Use Only</p>
    <p>Built with Streamlit, PyTorch, and Gymnasium</p>
</div>
""", unsafe_allow_html=True)

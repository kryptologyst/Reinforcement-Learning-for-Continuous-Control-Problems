# Reinforcement Learning for Continuous Control Problems

Research-ready implementation of reinforcement learning algorithms for continuous control tasks, featuring DDPG (Deep Deterministic Policy Gradient) and SAC (Soft Actor-Critic) algorithms.

## ⚠️ Safety Notice

**This is a research/educational demonstration only.** Do not use these algorithms for production control of real-world systems without extensive safety validation and risk assessment. These implementations are designed for learning and research purposes.

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Reinforcement-Learning-for-Continuous-Control-Problems.git
cd Reinforcement-Learning-for-Continuous-Control-Problems
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train an agent:
```bash
python -m src.train.train --env Pendulum-v1 --algorithm ddpg --episodes 1000
```

4. Evaluate the trained model:
```bash
python -m src.eval.eval --env Pendulum-v1 --algorithm ddpg --model checkpoints/best_model.pth --episodes 100
```

5. Launch the interactive demo:
```bash
streamlit run demo/app.py
```

## 📁 Project Structure

```
├── src/
│   ├── algorithms/          # RL algorithm implementations
│   │   ├── ddpg.py         # DDPG implementation
│   │   └── sac.py          # SAC implementation
│   ├── train/              # Training scripts
│   │   └── train.py        # Main training script
│   ├── eval/               # Evaluation scripts
│   │   └── eval.py         # Evaluation and analysis
│   └── utils/              # Utility functions
│       └── utils.py        # Common utilities
├── configs/                # Configuration files
│   ├── default.yaml        # Default configuration
│   └── sac.yaml           # SAC-specific configuration
├── demo/                   # Interactive demo
│   └── app.py             # Streamlit demo application
├── scripts/               # Helper scripts
├── tests/                 # Unit tests
├── assets/                # Generated plots and results
├── checkpoints/           # Saved model checkpoints
├── data/                  # Data storage
├── notebooks/             # Jupyter notebooks
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Supported Algorithms

### DDPG (Deep Deterministic Policy Gradient)
- **Type**: Off-policy, model-free
- **Best for**: Continuous control with deterministic policies
- **Key features**: Experience replay, target networks, OU noise exploration
- **Use case**: Robotic control, autonomous vehicles

### SAC (Soft Actor-Critic)
- **Type**: Off-policy, model-free
- **Best for**: Continuous control with stochastic policies
- **Key features**: Maximum entropy RL, automatic entropy tuning, twin Q-networks
- **Use case**: Complex continuous control tasks, exploration-heavy environments

## Supported Environments

- **Pendulum-v1**: Classic pendulum swing-up task
- **MountainCarContinuous-v0**: Continuous mountain car
- **BipedalWalker-v3**: Bipedal walking simulation
- **HalfCheetah-v4**: MuJoCo locomotion task
- **Ant-v4**: MuJoCo ant locomotion

## Usage

### Training

#### Basic Training
```bash
python -m src.train.train --env Pendulum-v1 --algorithm ddpg --episodes 1000
```

#### Advanced Training with Configuration
```bash
python -m src.train.train --env Pendulum-v1 --algorithm sac --config configs/sac.yaml --episodes 2000 --wandb
```

#### Training Parameters
- `--env`: Environment name (default: Pendulum-v1)
- `--algorithm`: Algorithm to use (ddpg/sac)
- `--episodes`: Number of training episodes (default: 1000)
- `--seed`: Random seed (default: 42)
- `--device`: Device to use (auto/cuda/mps/cpu)
- `--wandb`: Enable Weights & Biases logging
- `--config`: Path to configuration file

### Evaluation

#### Basic Evaluation
```bash
python -m src.eval.eval --env Pendulum-v1 --algorithm ddpg --model checkpoints/best_model.pth
```

#### Advanced Evaluation
```bash
python -m src.eval.eval --env Pendulum-v1 --algorithm sac --model checkpoints/best_model.pth --episodes 100 --render --save-plots
```

#### Evaluation Parameters
- `--env`: Environment name
- `--algorithm`: Algorithm used
- `--model`: Path to trained model
- `--episodes`: Number of evaluation episodes (default: 100)
- `--render`: Render the environment
- `--save-plots`: Save evaluation plots
- `--seed`: Random seed for evaluation

### Interactive Demo

Launch the Streamlit demo for interactive evaluation and visualization:

```bash
streamlit run demo/app.py
```

The demo provides:
- Real-time performance metrics
- Interactive trajectory visualization
- Action analysis and statistics
- Episode-by-episode results
- Downloadable data export

## Evaluation Metrics

### Performance Metrics
- **Average Return**: Mean episode reward ± 95% confidence interval
- **Success Rate**: Percentage of episodes reaching target performance
- **Sample Efficiency**: Episodes/steps to reach threshold performance
- **Stability**: Variance in performance across episodes

### Control Metrics
- **Action Smoothness**: Variance in action values
- **Exploration Efficiency**: State space coverage
- **Convergence Speed**: Learning curve analysis

### Safety Metrics (when applicable)
- **Constraint Violations**: Frequency of safety constraint breaches
- **Risk Measures**: CVaR and tail risk analysis

## Configuration

### Default Configuration (`configs/default.yaml`)
```yaml
env_name: "Pendulum-v1"
algorithm: "ddpg"
seed: 42

agent:
  actor_lr: 1e-4
  critic_lr: 1e-3
  gamma: 0.99
  tau: 0.005
  batch_size: 256
  buffer_size: 1000000
  hidden_dims: [256, 256]

training:
  num_episodes: 1000
  eval_freq: 100
  save_freq: 500
  eval_episodes: 10
```

### SAC Configuration (`configs/sac.yaml`)
```yaml
algorithm: "sac"
agent:
  lr: 3e-4
  alpha: 0.2
  auto_entropy: true
  # ... other parameters
```

## Reproducibility

### Deterministic Seeding
All random seeds are properly set for:
- Python random module
- NumPy random state
- PyTorch random state
- Environment random state

### Device Handling
Automatic device selection with fallback:
1. CUDA (if available)
2. MPS (Apple Silicon)
3. CPU (fallback)

### Checkpointing
- Automatic model saving at specified intervals
- Best model preservation based on evaluation performance
- Complete state saving (networks, optimizers, statistics)

## Expected Performance

### Pendulum-v1 Environment
| Algorithm | Mean Reward | Episodes to Converge | Notes |
|-----------|-------------|---------------------|-------|
| DDPG      | -150 ± 50   | ~500-800            | Stable, deterministic |
| SAC       | -200 ± 100  | ~300-600            | Better exploration |

*Performance may vary based on hyperparameters and random seeds*

## Testing

Run the test suite:
```bash
pytest tests/
```

Run specific tests:
```bash
pytest tests/test_ddpg.py
pytest tests/test_sac.py
```

## Algorithm Details

### DDPG Architecture
- **Actor Network**: Maps states to deterministic actions
- **Critic Network**: Estimates Q-values for state-action pairs
- **Target Networks**: Soft-updated copies for stability
- **Experience Replay**: Random sampling from past experiences
- **OU Noise**: Exploration through Ornstein-Uhlenbeck process

### SAC Architecture
- **Policy Network**: Stochastic policy with Gaussian distribution
- **Twin Q-Networks**: Two Q-networks to reduce overestimation bias
- **Target Q-Networks**: Soft-updated copies for stability
- **Entropy Regularization**: Automatic entropy coefficient tuning
- **Reparameterization Trick**: Differentiable action sampling

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Use CPU device: `--device cpu`

2. **Slow Training**
   - Enable GPU acceleration: `--device cuda`
   - Reduce buffer size for smaller environments

3. **Poor Performance**
   - Check hyperparameters in configuration files
   - Ensure proper environment setup
   - Verify model loading

4. **Import Errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version (3.10+ recommended)

### Performance Tips

1. **Hyperparameter Tuning**
   - Learning rates: Start with 1e-4 for actor, 1e-3 for critic
   - Network size: [256, 256] works well for most tasks
   - Buffer size: Larger for complex environments

2. **Environment Selection**
   - Start with Pendulum-v1 for algorithm validation
   - Use MuJoCo environments for complex locomotion
   - Consider environment-specific reward scaling

3. **Training Strategy**
   - Use evaluation-based early stopping
   - Monitor learning curves for convergence
   - Save checkpoints regularly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Setup
```bash
pip install -r requirements.txt
pip install -e .
pre-commit install
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gymnasium for the environment framework
- PyTorch for the deep learning framework
- The RL research community for algorithm implementations
- Contributors and users for feedback and improvements

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review the troubleshooting section

---

**Remember**: This is for research and educational purposes only. Always validate algorithms thoroughly before considering real-world applications.
# Reinforcement-Learning-for-Continuous-Control-Problems

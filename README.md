# Satellite Sensor Tasking with Reinforcement Learning

An educational Jupyter notebook demonstrating reinforcement learning applied to satellite sensor tasking, featuring Q-learning, transfer learning, and 3D visualization.

## Learning Objectives

By working through this project, you will:

1. **Understand RL Fundamentals**: Learn about states, actions, rewards, Q-learning, and epsilon-greedy exploration
2. **Apply RL to Real Problems**: See how RL can solve satellite sensor tasking challenges
3. **Visualize Agent Learning**: Track convergence with plots and heatmaps
4. **Explore Transfer Learning**: Understand how knowledge transfers between grid sizes
5. **Connect Theory to Practice**: Map grid-based learning to physical satellite gimbal pointing

## Features

- **Gymnasium-Compatible Environment**: Standard RL environment for satellite sensor tasking
- **Q-Learning Agent**: Tabular Q-learning with epsilon-greedy exploration
- **Transfer Learning**: Resize Q-tables to transfer knowledge between grid sizes (11x11 → 23x23)
- **2D Visualizations**: Learning curves, value function heatmaps, policy arrows
- **3D Satellite Visualization**: Gimbal pointing visualization with Earth, satellite, and targets
- **Comprehensive Tests**: Unit tests for all core modules
- **Educational Notebook**: Step-by-step Jupyter notebook with explanations

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Jupyter Lab or Jupyter Notebook

### Installation (< 10 minutes)

1. **Clone or download this repository**

2. **Navigate to project directory**:
   ```bash
   cd satellite_rl
   ```

3. **Create virtual environment**:
   ```bash
   # On Linux/macOS
   python3 -m venv venv_linux
   source venv_linux/bin/activate

   # On Windows
   python -m venv venv_windows
   venv_windows\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Install Jupyter kernel**:
   ```bash
   python -m ipykernel install --user --name=satellite_rl
   ```

6. **Launch Jupyter Lab**:
   ```bash
   jupyter lab
   ```

7. **Open the main notebook**:
   - Navigate to `notebook/satellite_sensor_tasking.ipynb`
   - Select Kernel → Change Kernel → satellite_rl
   - Run all cells (Cell → Run All)

## Project Structure

```
satellite_rl/
├── notebook/
│   └── satellite_sensor_tasking.ipynb   # Main educational notebook
├── src/
│   ├── environment/
│   │   └── grid_env.py                  # Gymnasium grid environment
│   ├── agents/
│   │   └── q_learning.py                # Q-learning agent
│   ├── visualization/
│   │   ├── grid_viz.py                  # 2D visualizations
│   │   └── satellite_viz.py             # 3D satellite visualizations
│   └── utils/
│       ├── helpers.py                   # Helper functions
│       └── transfer_learning.py         # Q-table resizing utilities
├── tests/
│   ├── test_environment.py              # Environment tests
│   ├── test_agents.py                   # Agent tests
│   └── test_transfer_learning.py        # Transfer learning tests
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## Usage

### Running the Main Notebook

The primary way to interact with this project is through the Jupyter notebook:

```bash
# Activate virtual environment
source venv_linux/bin/activate  # or venv_windows\Scripts\activate on Windows

# Launch Jupyter Lab
jupyter lab

# Open notebook/satellite_sensor_tasking.ipynb
# Select satellite_rl kernel
# Run cells sequentially or Run All
```

The notebook demonstrates:
1. Environment creation and exploration
2. Q-learning agent training (11x11 grid)
3. Learning convergence visualization
4. Value function and policy visualization
5. Transfer learning to larger grid (23x23)
6. 3D satellite gimbal visualization

### Using the Modules Programmatically

You can also use the modules directly in Python:

```python
from src.environment.grid_env import SatelliteSensorGridEnv
from src.agents.q_learning import QLearningAgent
from src.visualization.grid_viz import plot_learning_curve

# Create environment
env = SatelliteSensorGridEnv(grid_x=11, grid_y=11)

# Create agent
agent = QLearningAgent(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)

# Train
scores = agent.train(num_episodes=30000, max_steps=50)

# Visualize
plot_learning_curve(scores, window=500)
```

## Running Tests

```bash
# Activate virtual environment
source venv_linux/bin/activate

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_environment.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Key Concepts

### Environment

- **Grid-based abstraction**: Satellite sensor tasking represented as navigation on an odd-sized grid (11x11, 23x23)
- **State**: Current sensor pointing position (integer 0 to num_states-1)
- **Actions**: Up (0), Down (1), Left (2), Right (3)
- **Goal**: Center of grid (high-value target)
- **Rewards**: +100 for reaching goal, 0 otherwise

### Q-Learning

The agent learns a Q-table Q(s,a) representing expected cumulative rewards:

```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

Where:
- α (alpha) = learning rate (0.1)
- γ (gamma) = discount factor (0.9)
- r = immediate reward
- s' = next state

### Transfer Learning

Knowledge learned on a small grid (11x11) is transferred to a larger grid (23x23) by:

1. Training on 11x11 grid
2. Extracting Q-table (max Q-value per state)
3. Resizing Q-table using midpoint interpolation
4. Initializing 23x23 agent with resized values
5. Training faster due to prior knowledge

## Troubleshooting

### Kernel Not Found

If the satellite_rl kernel doesn't appear:

```bash
python -m ipykernel install --user --name=satellite_rl
jupyter kernelspec list  # Verify it's installed
```

### Import Errors

Ensure you're in the virtual environment:

```bash
which python  # Should show venv path
pip list  # Verify packages installed
```

### Visualization Not Displaying

For 3D plots in Jupyter:

```python
%matplotlib inline  # For static plots
# or
%matplotlib notebook  # For interactive plots (may not work in JupyterLab)
```

### Slow Training

Training 30,000+ episodes can take a few minutes. Reduce `num_episodes` for faster testing:

```python
scores = agent.train(num_episodes=1000, max_steps=50)  # Faster
```

## Dependencies

Core dependencies (see `requirements.txt` for full list):

- **numpy**: Numerical computations and Q-table operations
- **matplotlib**: 2D/3D visualizations
- **gymnasium**: RL environment framework
- **jupyter/jupyterlab**: Interactive notebook interface
- **pytest**: Unit testing framework
- **black/ruff**: Code formatting and linting
- **mypy**: Static type checking

## Further Exploration

### For Students

1. **Experiment with hyperparameters**: Try different learning rates (α), discount factors (γ), and exploration rates (ε)
2. **Modify reward structure**: Add distance-based rewards or penalties for inefficient paths
3. **Multi-target scenarios**: Extend to multiple targets with different priorities
4. **Compare algorithms**: Implement SARSA or Deep Q-Learning
5. **Realistic orbits**: Integrate with poliastro library for true orbital mechanics

### Advanced Topics

- **Multi-agent coordination**: Multiple satellites coordinating sensor tasking
- **Continuous action spaces**: Continuous gimbal angles instead of discrete grid
- **Partial observability**: Limited sensor field of view
- **Dynamic targets**: Moving or time-varying targets

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2018)
- Gymnasium Documentation: https://gymnasium.farama.org/
- Satellite RL Research: https://arxiv.org/html/2409.02270v1

## Contributing

This is an educational project. Feel free to:

- Report issues or bugs
- Suggest improvements
- Add new features or examples
- Improve documentation

## License

This project is provided for educational purposes.

## Acknowledgments

- Built with Gymnasium (modern OpenAI Gym)
- Inspired by satellite sensor tasking research
- Educational design following best practices from RL literature

---

**Need Help?** Check the troubleshooting section or review the comprehensive docstrings in the source code.

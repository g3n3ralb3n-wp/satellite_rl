# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is an **educational reinforcement learning project** demonstrating satellite sensor tasking using Q-learning. The codebase includes a Gymnasium-compatible environment, a tabular Q-learning agent, transfer learning utilities, and both 2D and 3D visualization tools. The primary interface is a Jupyter notebook that walks through the concepts step-by-step.

## Development Environment

### Virtual Environment
- **Always use `venv_linux`** for all Python commands (on Linux/WSL)
- **Windows users**: use `venv_windows` instead
- Activate before running any Python code, tests, or Jupyter:
  ```bash
  source venv_linux/bin/activate  # Linux/macOS
  venv_windows\Scripts\activate   # Windows
  ```

### Essential Commands

**Running Tests:**
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_environment.py -v

# Single test
pytest tests/test_environment.py::test_environment_reset -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

**Code Quality:**
```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

**Running Jupyter:**
```bash
jupyter lab
# Then open notebook/satellite_sensor_tasking.ipynb
# Select kernel: satellite_rl
```

## Code Architecture

### Core Components

The project follows a modular structure with clear separation of concerns:

1. **Environment** ([src/environment/grid_env.py](src/environment/grid_env.py))
   - `SatelliteSensorGridEnv`: Gymnasium-compatible environment
   - Grid-based abstraction of satellite sensor tasking
   - **Key constraint**: Grid dimensions must be odd (forces center goal position)
   - Discrete state space (0 to num_states-1) and action space (4 directions)
   - Reward matrix encodes boundaries (NaN) and goal-adjacent rewards (+100)

2. **Agent** ([src/agents/q_learning.py](src/agents/q_learning.py))
   - `QLearningAgent`: Tabular Q-learning with epsilon-greedy exploration
   - Supports warm-start via `initial_q_table` parameter for transfer learning
   - Q-table shape: `(num_states, num_actions)`
   - Training returns convergence scores (sum of Q-table values)

3. **Transfer Learning** ([src/utils/transfer_learning.py](src/utils/transfer_learning.py))
   - Midpoint interpolation for resizing Q-tables (11x11 → 23x23)
   - Bidirectional conversion between Q-table (2D values) and Q-matrix (states × actions)
   - **Critical pattern**: `resize_one_array` uses iterative midpoint insertion
   - Knowledge transfer workflow:
     1. Train small agent (11×11)
     2. Extract Q-table: `agent.get_value_grid()`
     3. Resize: `resize_q_table(q_table, 23, 23)`
     4. Convert to Q-matrix: `q_table_to_q_matrix(resized_table)`
     5. Initialize large agent with resized Q-matrix

4. **Visualization** ([src/visualization/](src/visualization/))
   - `grid_viz.py`: 2D plots (learning curves, heatmaps, policy arrows)
   - `satellite_viz.py`: 3D gimbal visualization (Earth, satellite, target)

### Data Flow

```
Environment.reset() → state (random position)
    ↓
Agent.select_action(state) → action (epsilon-greedy)
    ↓
Environment.step(action) → next_state, reward, terminated
    ↓
Agent updates Q-table: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
    ↓
Repeat until terminated or max_steps
```

### Design Patterns

- **Odd grid enforcement** ([grid_env.py:50-54](src/environment/grid_env.py#L50-L54)): Grid dimensions are forced odd to ensure a unique center goal position
- **Reward matrix with NaN boundaries** ([grid_env.py:79-113](src/environment/grid_env.py#L79-L113)): Invalid actions encoded as NaN for clean boundary handling
- **State indexing** ([grid_env.py:169-175](src/environment/grid_env.py#L169-L175)): 1D state index = row × grid_y + col
- **Convergence tracking**: Training returns sum of all Q-values as convergence metric (not episode rewards)

## Testing Philosophy

- **Every module has corresponding tests** in `tests/` directory
- **Test structure mirrors source structure**: `src/environment/grid_env.py` → `tests/test_environment.py`
- **Minimum test coverage**: expected use, edge case, failure case
- Tests use small grids (3×3, 5×5) for fast execution

## Key Gotchas

1. **Grid dimensions must be odd** - Environment enforces this automatically but be aware when designing tests or extensions
2. **State vs. position** - State is 1D index (0 to n-1), position is 2D tuple (row, col)
3. **Q-table vs. Q-matrix**:
   - Q-table: 2D grid of max values per state (for visualization)
   - Q-matrix: 2D array of shape (num_states, num_actions) (for agent)
4. **Convergence score** - Training returns sum of Q-values, NOT episode rewards
5. **Transfer learning direction** - Resizing only works small → large (upsampling)
6. **Virtual environment** - Must use `venv_linux` or `venv_windows`, not base Python

## Dependencies

Core libraries (see [requirements.txt](requirements.txt)):
- `gymnasium>=0.29.0` - RL environment framework (successor to OpenAI Gym)
- `numpy>=1.24.0` - Q-table operations and numerical computation
- `matplotlib>=3.7.0` - Visualization
- `pytest>=7.4.0` - Testing
- `jupyter/jupyterlab` - Educational notebook interface
- `black`, `ruff`, `mypy` - Code quality tools

## Extending the Project

Common extension points:

- **New environments**: Subclass `gym.Env`, follow Gymnasium API
- **New agents**: Implement `select_action()` and `train()` methods
- **New visualizations**: Add to `src/visualization/`, use same style (matplotlib)
- **New transfer learning methods**: Add to `src/utils/transfer_learning.py`

When adding features:
1. Create implementation in `src/`
2. Add tests in `tests/`
3. Update notebook if relevant to educational flow
4. Run full test suite to ensure no regressions

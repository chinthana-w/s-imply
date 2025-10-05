# Reverse Circuit Simulator

A transformer-based neural network system that learns to predict input patterns for digital circuits given functional embeddings and desired output values. This system combines deep learning with circuit simulation for automated test pattern generation and circuit analysis.

## Overview

The Reverse Circuit Simulator addresses the problem of **reverse engineering circuit behavior**: given a circuit's functional representation and a desired output, predict the input pattern that would produce that output. This is particularly useful for:

- **Automated Test Pattern Generation (ATPG)**
- **Circuit verification and validation**
- **Fault diagnosis and debugging**
- **Circuit optimization**

## System Architecture

### Core Components

1. **Transformer Model** (`reverse_simulator.py`)
   - Multi-head attention mechanism for processing functional embeddings
   - Handles variable-length input sequences
   - Outputs binary predictions for input patterns

2. **Training Environment** (`training_env.py`)
   - Gymnasium-compatible RL environment
   - Integrates with existing circuit simulation infrastructure
   - Provides reward signals based on simulation results

3. **Training Pipeline** (`trainer.py`)
   - Supervised learning with generated training data
   - Reinforcement learning with circuit simulation feedback
   - Comprehensive evaluation and checkpointing

4. **Embedding System** (existing `gcn.py`)
   - Uses DeepGate for functional and structural embeddings
   - 128-dimensional functional embeddings for each circuit node

## Data Flow

```
Circuit (.bench) → DeepGate → Functional Embeddings → Transformer → Input Pattern → Circuit Simulation → Reward
```

### Input Format
- **Functional Embeddings**: 128-dimensional vectors representing circuit nodes
- **Desired Output**: Binary value (0 or 1) for the target output
- **Circuit Structure**: Parsed from .bench files

### Output Format
- **Binary Vector**: Predicted input pattern (0s and 1s)
- **Confidence Scores**: Model confidence for each prediction

## Installation and Setup

### Prerequisites
```bash
# Required Python packages
pip install torch torchvision
pip install gymnasium
pip install numpy matplotlib tqdm
pip install deepgate  # For circuit embeddings
```

### Directory Structure
```
src/ml/
├── reverse_simulator.py    # Transformer model
├── training_env.py         # RL environment
├── trainer.py             # Training utilities
├── example_usage.py       # Usage examples
└── gcn.py                 # Embedding extraction (existing)
```

## Usage Examples

### 1. Single Circuit Prediction

```python
from src.ml.reverse_simulator import create_model
from src.ml.gcn import bench_to_embed

# Load circuit and extract embeddings
struct_emb, func_emb = bench_to_embed("circuit.bench")

# Create model
model = create_model(device='cuda')

# Prepare inputs
input_embeddings = func_emb[:num_inputs]  # Input node embeddings
output_embedding = func_emb[num_inputs:]  # Output node embedding
desired_output = torch.tensor([[1]])      # Desired output value

# Get prediction
prediction = model(input_embeddings, output_embedding, desired_output)
binary_pattern = (prediction > 0.5).float()
```

### 2. Training Environment

```python
from src.ml.training_env import create_training_env

# Create environment with circuit pool
circuit_pool = glob.glob("data/bench/*.bench")
env = create_training_env(circuit_pool)

# Run episode
obs, info = env.reset()
action = env.predict_action(obs)
next_obs, reward, done, truncated, info = env.step(action)
```

### 3. Model Training

```python
from src.ml.trainer import ReverseSimulatorTrainer

# Create trainer
trainer = ReverseSimulatorTrainer(
    model=model,
    circuit_pool=circuit_pool,
    learning_rate=1e-4,
    batch_size=32
)

# Supervised training
trainer.train_supervised(num_epochs=100, batch_size=32)

# Reinforcement learning
trainer.train_reinforcement(num_episodes=1000)
```

## Model Architecture Details

### Transformer Configuration
- **Embedding Dimension**: 128 (matches DeepGate output)
- **Model Dimension**: 256
- **Attention Heads**: 8
- **Layers**: 6
- **Feedforward Dimension**: 1024
- **Max Inputs**: 100 (configurable)

### Input Sequence Structure
```
[Output_Token, Output_Embedding, Separator_Token, Input_Embedding_1, ..., Input_Embedding_N]
```

### Loss Function
Combines binary cross-entropy with simulation feedback:
```
Loss = α × BCE_Loss + β × Simulation_Reward_Loss
```

## Training Strategies

### 1. Supervised Learning
- Uses generated training data with known input-output pairs
- Binary cross-entropy loss for pattern prediction
- Validation on held-out circuits

### 2. Reinforcement Learning
- Direct circuit simulation feedback
- Reward: +1 for correct output, -1 for incorrect
- Policy gradient updates based on simulation results

### 3. Hybrid Approach
- Pre-train with supervised learning
- Fine-tune with reinforcement learning
- Best of both worlds: data efficiency + simulation accuracy

## Evaluation Metrics

- **Prediction Accuracy**: Percentage of correctly predicted input patterns
- **Simulation Success Rate**: Percentage of predictions that produce correct circuit output
- **Confidence Calibration**: How well model confidence correlates with actual accuracy
- **Circuit Coverage**: Performance across different circuit types and sizes

## Circuit Compatibility

The system works with circuits in .bench format:
- **ISCAS85**: Standard benchmark circuits (c17, c432, c880, etc.)
- **ISCAS89**: Larger benchmark circuits
- **Custom circuits**: Any .bench format circuit

### Supported Gate Types
- Primary inputs/outputs
- AND, NAND, OR, NOR gates
- XOR, XNOR gates
- NOT gates, buffers

## Performance Considerations

### Computational Requirements
- **GPU Memory**: ~2-4GB for training (depends on batch size)
- **Training Time**: 1-4 hours for 100 epochs (depends on circuit pool size)
- **Inference Speed**: ~1ms per prediction on GPU

### Optimization Tips
1. **Batch Processing**: Use larger batch sizes for better GPU utilization
2. **Circuit Pool**: Include diverse circuit types for better generalization
3. **Learning Rate**: Start with 1e-4, use learning rate scheduling
4. **Gradient Clipping**: Prevents exploding gradients in transformer

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Process circuits in smaller groups

2. **Poor Convergence**
   - Check learning rate (try 1e-5 to 1e-3)
   - Increase model capacity (more layers/heads)
   - Add more diverse training circuits

3. **Simulation Errors**
   - Verify circuit format compatibility
   - Check gate type support
   - Ensure proper input/output identification

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check embedding shapes
print(f"Input embeddings: {input_embeddings.shape}")
print(f"Output embedding: {output_embedding.shape}")
```

## Future Enhancements

1. **Multi-Output Circuits**: Support circuits with multiple outputs
2. **Sequential Circuits**: Handle flip-flops and sequential logic
3. **Fault Injection**: Train on faulty circuits for robust testing
4. **Transfer Learning**: Pre-train on large circuit datasets
5. **Attention Visualization**: Understand which circuit parts the model focuses on

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all linting passes
5. Submit a pull request

## License

This project is part of the s-imply ATPG system. See the main LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{reverse_circuit_simulator,
  title={Transformer-based Reverse Circuit Simulator for ATPG},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/s-imply}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

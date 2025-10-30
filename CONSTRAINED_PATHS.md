# Constrained Path Assignments

## Overview

This enhancement adds support for learning and predicting "constrained" path pair assignments. The system can now set up scenarios where some gates already have predetermined values (0 or 1), and the model must predict assignments compatible with these constraints.

## Key Changes

### 1. Logic Value Feature in Embeddings

**Location**: `src/ml/reconv_ds.py`

The node embeddings now include an additional dimension for the logic value:
- **Original**: `[P, L, D]` where D=128
- **New**: `[P, L, D+1]` where the last dimension encodes the logic value

Logic value encoding:
- `0.0` = Node constrained to logic 0
- `1.0` = Node constrained to logic 1  
- `2.0` = Node unconstrained (X/don't care)

### 2. Constraint Generation

**Location**: `src/ml/reconv_ds.py` - `_generate_constraints()` method

For each reconvergent path sample, constraints are randomly generated:

1. **Start nodes**: All first nodes get a shared random value (0 or 1)
2. **Reconvergence point**: All last nodes get a shared random value (0 or 1)
3. **Intermediate nodes**: ~25% get random values (0 or 1)
4. **Remaining nodes**: Unconstrained (value 2)

This ensures:
- Constraints are valid (start and end points match across paths)
- Model sees diverse scenarios (both constrained and unconstrained)
- Training is more challenging (model must respect existing values)

### 3. Model Input Projection

**Location**: `src/ml/reconv_lib.py` - `MultiPathTransformer` class

Added automatic input projection to handle dimension compatibility:
- Input: 129 dimensions (128 + 1)
- Projected: 132 dimensions (nearest multiple of nhead=4)
- This ensures compatibility with the transformer's multi-head attention

### 4. Training Configuration

**Location**: `src/ml/train_reconv.py`

Updated model instantiation to use `embedding_dim + 1` instead of `embedding_dim`.

## Usage

No changes to command-line arguments or workflow! The system automatically:

1. Generates constraints when loading dataset samples
2. Appends logic values to embeddings
3. Projects to compatible dimensions

Training command remains the same:
```bash
python -m src.ml.train_reconv train \
  --dataset data/datasets/reconv_dataset.pkl \
  --output checkpoints/reconv_minimal \
  --epochs 10
```

## Example

For a reconvergent structure with two paths:
- Path 1: `[node_1, node_2, node_3, node_4, node_5]`
- Path 2: `[node_1, node_6, node_7, node_8, node_5]`

A sample constraint scenario:
```
Node 1 (start):           value = 1  (constrained)
Node 2 (intermediate):    value = 2  (unconstrained)
Node 3 (intermediate):    value = 0  (constrained)
Node 4 (intermediate):    value = 2  (unconstrained)
Node 5 (reconverge):      value = 0  (constrained)
Node 6 (intermediate):    value = 2  (unconstrained)
Node 7 (intermediate):    value = 1  (constrained)
Node 8 (intermediate):    value = 2  (unconstrained)
```

The model receives embeddings with these values appended and must predict assignments that:
- Respect the constrained values (nodes 1, 3, 5, 7)
- Complete the unconstrained nodes (nodes 2, 4, 6, 8)
- Maintain logical consistency along the paths

## Benefits

1. **More realistic scenarios**: Real circuits often have partial assignments
2. **Improved learning**: Model learns to work with constraints
3. **Better generalization**: Model handles both constrained and unconstrained cases
4. **Minimal changes**: No new files, no new command-line arguments
5. **Backward compatible**: Works with existing datasets

## Technical Details

### Constraint Probability

The constraint probability is set to 0.25 (25%) by default. This can be adjusted in the `ReconvergentPathsDataset.__init__()` method via the `constraint_prob` parameter if needed.

### Dimension Handling

The model automatically handles dimension compatibility:
- If `embedding_dim % nhead != 0`, input projection is added
- Projects to nearest multiple of `nhead` 
- Example: 129 → 132 when `nhead=4`

### Constraint Validity

Constraints are guaranteed to be valid because:
- All start nodes share the same value
- All reconvergence points share the same value
- This matches the physical constraint that these nodes must have consistent values

## Testing

Run the test script to verify the implementation:
```bash
python test_constrained_paths.py
```

This validates:
- Constraint generation logic
- Embedding dimensions
- Model compatibility
- Training workflow

# Differentiable SLIC PyTorch
This is a PyTorch implementation of differentiable SLIC.  
Basecally, this algorithm is based on Superpixel Sampling Networks (SSN) but slightly different:
1. Our diffSLIC computes similarity between pixels and superpixels by inner product. Because of it, we use the positional embedding instead of raw xy-coordinate in a test code.
2. Original SNN implementation computes $\tilde{Q}$ by normalizing row of $\hat{Q}$, but our implementation directly normalize $Q$.
3. Our implementation can adjust a neighborhood radius (i.e., the range of the candidate superpixels)

## Enviroment
- Python 3.8
- PyTorch 1.12
- NumPy 1.23

## How to use
See docstring of each function and class.
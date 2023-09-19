# Differentiable SLIC PyTorch
This is a PyTorch implementation of differentiable SLIC which computes superpixels with the _soft_ assignment.  
Unlike the original SLIC, the similarity between pixels and centers is computed by the inner product, which corresponds to the clustering step of HCFormer.

## Environment
- Python==3.8
- PyTorch==1.12
- NumPy==1.23

## How to use
See the docstring of each function and class for details.

Simple usage:
```python
import torch

from diffSLIC import DiffSLIC

slic_fn = DiffSLIC(n_spixels=100, n_iter=5, tau=0.01, candidate_radius=1, stable=True)

rgb_img = torch.arange(30000).reshape(1, 3, 100, 100)
features, spix2pix_assign, pix2spix_assign = slic_fn(rgb_img)
```

## Citation
This repository:
```
@misc{diffSLIC,
    title = {Differentiable SLIC},
    author = {Suzuki, Teppei},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/perrying/diffSLIC}},
    year = {2022},
}
```
HCFormer:
```
@article{suzuki2022clustering,
  title={Clustering as Attention: Unified Image Segmentation with Hierarchical Clustering},
  author={Suzuki, Teppei},
  journal={arXiv preprint arXiv:2205.09949},
  year={2022}
}
```
and its preliminary work:
```
@article{suzuki2021implicit,
  title={Implicit Integration of Superpixel Segmentation into Fully Convolutional Networks},
  author={Suzuki, Teppei},
  journal={arXiv preprint arXiv:2103.03435},
  year={2021}
}
```

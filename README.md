# Differentiable SLIC PyTorch
This is a PyTorch implementation of differentiable SLIC which computes superpixels with the _soft_ assignment.  
Unlike the original SLIC, the similarity between pixels and centers is computed by the inner product, which corresponds to the clustering step of HCFormer.

## Environment
- Python==3.8
- PyTorch==1.12
- NumPy==1.23

## How to use
See the docstring of each function and class.

## Citation
This repository:
```
@misc{diffSLIC,
    author = {Teppei Suzuki},
    title = {Differentiable SLIC},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/perrying/diffSLIC}},
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

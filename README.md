# Towards Model-Agnostic Dataset Condensation by Heterogeneous Models (ECCV 2024, Oral)

[![arXiv](https://img.shields.io/badge/arXiv-2409.14538-b31b1b.svg)](https://arxiv.org/abs/2409.14538)

Official PyTorch implementation for the ECCV 2024 paper:

**Towards Model-Agnostic Dataset Condensation by Heterogeneous Models**  
*Jun-Yeong Moon, Jung Uk Kim* $^\dagger$*, Gyeong-Moon Park* $^\dagger$

## Installation

We recommend using a Conda environment to manage dependencies. You can create the required environment by running:

```bash
conda create -f environment.yml
```
Alternatively, you can manually install the necessary packages:

- python>=3.10
- torch>=2.1.0
- timm
- matplotlib
- scikit-learn

## Dataset Condensation

To generate a condensed dataset, execute the following script:

```bash
./scripts/run_dual.sh CIFAR10 aug_kmeans 10 128 5e-3 ConvNet 0.01 ViT_Tiny_ft 0.001 ./PATH
```

- Note: All models except ConvNet should have either _ft (pretrained) or _scratch (random initialization) appended to their names.
You can find the implementations for ConvNet, ViG-ti, s, b in the models directory.
Additionally, any ResNet or Vision Transformer available in the timm library can be used as models.

ConvNet, ViG-ti, s, b is implemented in the `models` directory.

Also, ResNets, and Vision Transformers that is available in timm library can be used as a model.

## Evaluate the Condensed Dataset

To evaluate the condensed dataset, use the following command:

```bash
./scripts/run_test_condensation.sh CIFAR10 2000 128 ./PATH --ft
```
- Note: Append --ft or --scratch at the end of the command depending on the model type used (pretrained or randomly initialized).
- The PATH should include synthetic_images.pth

## Acknowledgements

This code is inspired by and builds upon several pioneering works, including:

- [Vision GNN (NeurIPS2022)](https://github.com/jichengyuan/Vision_GNN)
- [CAFE (CVPR2022)](https://github.com/kaiwang960112/CAFE)
- [MTT (CVPR2022)](https://github.com/GeorgeCazenavette/mtt-distillation)
- [IDC (ICML22)](https://github.com/GeorgeCazenavette/mtt-distillation)
- [IDM (CVPR2023)](https://github.com/GeorgeCazenavette/mtt-distillation)
- [DREAM (ICCV2023)](https://github.com/GeorgeCazenavette/mtt-distillation)

We are grateful to these authors and the wider research community for their contributions.

## Citation

```BibTeX
@misc{moon2024modelagnosticdatasetcondensationheterogeneous,
      title={Towards Model-Agnostic Dataset Condensation by Heterogeneous Models}, 
      author={Jun-Yeong Moon and Jung Uk Kim and Gyeong-Moon Park},
      year={2024},
      eprint={2409.14538},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.14538}, 
}
```
- Note: The citation information will be updated.


If you experience any problems or discover a bug, please feel free to reach out via email, submit an issue, or submit a pull request.

Your feedback is invaluable in helping us improve, and we will review and address the matter as promptly as possible.

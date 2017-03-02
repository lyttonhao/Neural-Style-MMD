# Neural-Style-MMD

This repository holds the MXNet code for the paper

>
**Demystifying Neural Style Transfer,
Yanghao Li, Naiyan Wang, Jiaying Liu, and Xiaodi Hou,
arXiv preprint arXiv:1701.01036
>
[[Arxiv Preprint](https://arxiv.org/abs/1701.01036)]


## Introduction

Neural-Style-MMD presents a neural style transfer algorithm based on a new interpretation. Instead of using Gram matrix in original neural style transfer methods, this repo provides two methods to implement style transfer, including a Maximum Mean Discrepancy (MMD) loss and a Batch Normalization (BN) statistic loss. The paper also demonstrates the original matching Gram matrix is equivalent to the a specific polynomial MMD. Details could be found in the paper. Our implementation is based on the [neural-style example](https://github.com/dmlc/mxnet/tree/master/example/neural-style) of MXNet.

## Prerequisites

Before running this code, you should make the following preparations:

* Install MXNet following the [instructions](http://mxnet.io/get_started/index.html#setup-and-installation) and install the python interface. This repo is tested on commmit 01cde1.

* Download the pre-trained VGG-19 model in the `model` folder:
```shell
wget https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/model/vgg19.params
```

## Usage

Basic Usage:
```shell
python neural-style.py --mmd-kernel linear --gpu 0 --style-weight 5.0 --content-image input/brad_pitt.jpg --style-image input/starry_night.jpg --output brad_pitt-starry_night --output-folder output_images
```
We support 4 single transfer methods, including 3 mmd kernels, including linear, poly and Gaussian, and a BN Statistics Matching method. At the same time, the code supports fusing different transfer methods with specific weights.

**Options
* `--mmd-kernel`: Specify MMD kernel (`linear`, `poly`, `Gaussian`), also their combination, e.g. `linear,poly`.
* `--bn-loss`: Whether to use the BN method. 
* `--multi-weights`: The weights when fusing different transfer methods, e.g. `0.5,0.5`.
* `--style-weight`: How much to weight the style loss term. It is equivalent to the balance factor gamma in the paper when we fix the `content-weight` as 1.0.

You can run `python neural-style.py` with `-h` to see more options.


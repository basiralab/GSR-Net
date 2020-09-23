# GSR-Net: Graph Super-Resolution Network

This repository provides the official PyTorch implementation of the following paper:

**Graph Super-Resolution Network for predicting high-resolution connectomes from low-resolution connectomes.**
[Megi Isallari](https://github.com/meg-i)<sup>1</sup>, [Islem Rekik](https://basira-lab.com/)<sup>1</sup>

> <sup>1</sup>BASIRA Lab, Faculty of Computer and Informatics, Istanbul Technical University, Istanbul, Turkey

Please contact isallari.megi@gmail.com for further inquiries. Thanks.

While a significant number of image super-resolution methods have been proposed for MRI super-resolution, building generative models for super-resolving a low-resolution brain connectome at a higher resolution
(i.e., adding new graph nodes/edges) remains unexplored —although this would circumvent the need for costly data collection and manual labelling of anatomical brain regions (i.e. parcellation). To fill this gap, we introduce GSR-Net (Graph Super-Resolution Network), the first super-resolution framework operating on graph-structured data that generates high-resolution brain graphs from low-resolution graphs.

![GSR-Net pipeline](/images/concept_fig.PNG)

# Detailed proposed GSR-Net pipeline

This work has been accepted to “Machine Learning in Medical Imaging” (MLMI) MICCAI 2020 workshop. The key idea of **GSR-Net (Graph Super-Resolution Network)** can be summarized in three fundamental steps: (i) learning feature embeddings for each brain ROI (node) in the LR connectome, (ii) the design of a graph super-resolution operation that predicts an HR connectome from the LR connectivity matrix and feature embeddings of the LR connectome computed in (i), (iii) learning node feature embeddings for each node in the super-resolved (HR) graph obtained in (ii). We used 5-fold cross-validation to evaluate our framework on 277 subjects from the Southwest University Longitudinal Imaging Multimodal (SLIM) study:
http://fcon_1000.projects.nitrc.org/indi/retro/southwestuni_qiu_index.html.
In this repository, we release the code for training and testing GSR-Net on the SLIM dataset.

![GSR-Net pipeline](/images/finalfig.png)

# Dependencies

The code has been tested with Google Colaboratory which uses Ubuntu 18.04.3 LTS Bionic,
Python 3.6.9 and PyTorch 1.4.0. In case you opt to run the code locally, you need to install the following python packages via pip:

- [Python 3.6+](https://www.python.org/)
- [PyTorch 1.4.0+](http://pytorch.org/)
- [Scikit-learn 0.22.2+](https://scikit-learn.org/stable/)
- [Matplotlib 3.2.2+](https://matplotlib.org/)
- [Numpy 1.18.5+](https://numpy.org/)

# Demo

We provide a demo code in `demo.py` to run the script of GSR-Net for predicting high-resolution connectomes from low-resolution functional brain connectomes. To set the parameters, you should provide commandline arguments.

If you want to run the code in the hyperparameters described in the paper, you can run it without any commandline arguments:

```sh
$ python demo.py
```

It would be equivalent to:

```sh
$ python demo.py –epochs=200 –lr=0.0001 –splits=5 –lmbda=16 –lr_dim=160 –hr_dim=320 –hidden_dim=320 –padding=26
```

To learn more about how to use the arguments:

```sh
$ python demo.py --help
```

| Plugin     | README                                         |
| ---------- | ---------------------------------------------- |
| epochs     | number of epochs to train                      |
| lr         | learning rate of Adam Optimizer                |
| splits     | number of cross validation folds               |
| lmbda      | self-reconstruction error hyper-parameter      |
| lr_dim     | number of nodes of low-resolution brain graph  |
| hr_dim     | number of nodes of high-resolution brain graph |
| hidden_dim | number of hidden GCN layer neurons             |
| padding    | dimensions of padding                          |

**Data preparation**

In our paper, we have used the SLIM dataset. In this repository, we simulated a n x l x l tensor X ( low-resolution connectomes for all subjects where l is the number of nodes of the LR connectome) and a n x h x h tensor Y (high-resolution connectomes for all subjects where h is the number of nodes of the HR connectome) It might yield in suboptimal results since data is randomly generated opposed to real brain graph data.

To use a dataset of your own preference, you can edit the data() function at preprocessing.py. In order to train and test the framework, you need to provide:

1. `N` low-resolution brain graph connectomes of dimensions `L*L` for variable `X` in `demo.py`
2. `N` high-resolution brain graph connectomes of dimensions `H*H` for variable `Y` in `demo.py`

# Example Results

If you run the demo with the default parameter setting as in the command below,

```sh
$ python demo.py –epochs=200 –lr=0.0001 –splits=5 –lmbda=16 –lr_dim=160 –hr_dim=320 –hidden_dim=320 –padding=26
```

you will get the following outputs:

![GSR-Net pipeline](/images/example.png)


# YouTube videos to install and run the code and understand how GSR-Net works

To install and run GSR-Net, check the following YouTube video:
https://youtu.be/xwHKRxgMaEM

To learn about how GSR-Net works, check the following YouTube video:
https://youtu.be/GahVu9NeOIg

# Related references

Graph U-Nets: Gao, H., Ji, S.: Graph u-nets. In Chaudhuri, K., Salakhutdinov, R., eds.: Proceedings of the
36th International Conference on Machine Learning. Volume 97 of Proceedings of Machine Learning Research., Long Beach, California, USA, PMLR (2019) 2083–2092 [https://github.com/HongyangGao/Graph-U-Nets]

SLIM Dataset: Liu, W., Wei, D., Chen, Q., Yang, W., Meng, J., Wu, G., Bi, T., Zhang, Q., Zuo, X.N., Qiu,
J.: Longitudinal test-retest neuroimaging data from healthy young adults in southwest china. Scientific Data 4 (2017) [https://www.nature.com/articles/sdata201717]

# Please cite the following paper when using GSR-Net:

```latex
@inproceedings{Ilassari2020,
title={Graph Super-Resolution Network for predicting high-resolution connectomes from low-resolution connectomes},
author={Isallari, Megi and Rekik, Islem},
booktitle={International Workshop on PRedictive Intelligence In MEdicine},
year={2020},
organization={Springer}
}
```

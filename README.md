# neural-ot

This repository contains experiments from the original paper by [Seguy et al, 2018](https://arxiv.org/pdf/1711.02283.pdf). The authors of the paper presented a novel two-step approach for the fundamental problem of learning an optimal map from one distribution to another.

## What's inside?

Here you can find a brief description of the experiments implemented in this repository.

### Toy Experiment

<p align="center">
  <img width="500" alt="Toy Experiment 1" src="https://github.com/Daniil-Selikhanovych/neural-ot/blob/master/img/gauss1.png?raw=true">
</p>
<p align="center">
  <img width="500" alt="Toy Experiment 2" src="https://github.com/Daniil-Selikhanovych/neural-ot/blob/master/img/gauss2.png?raw=true">
</p>

In this experiment we learn to **transform** samples from the unimodal Multivariate Normal (**continuous distribution**) into the pre-computed samples from multimodal mixture of other Multivariate Normals (**discrete distribution**). We solve an Optimal Transport problem by representing the optimal mapping with neural network. Then we check that the original density has been successfully transformed.

### Domain Adaptation

<p align="center">
  <img width="500" alt="Domain Adaptation" src="https://github.com/Daniil-Selikhanovych/neural-ot/blob/master/img/mappings.png?raw=true">
</p>

It is a widespread problem to have two similar datasets, but be able to train a classifier on only one of them, let's call it *target*. The other dataset will be called *source*. We can deal with this problem by computing an **optimal mapping** between the *source* and the *target*. Now during training we can use **mapped samples** from the *source* dataset in addition to *target* dataset samples. We test this assumtion by training *1-NN* on the untransformed *source* and then — on the transformed samples from *source*. In our setting we study **MNIST** to **USPS** and **USPS** to **MNIST** mappings.

### Generative Modeling

<p align="center">
  <img width="500" alt="Generative Modeling" src="https://github.com/Daniil-Selikhanovych/neural-ot/blob/master/img/generated.png?raw=true">
</p>

Suppose, we have computed an **optimal transport** between some simple distribution (Multivariate Normal, for example) into a much more complex one. In our case we work with the digits from the **MNIST** dataset. We can then use the optimal mapping to **generate** new **MNIST**-like samples. One can first sample several points from the **Multivariate Normal** and then transform them into the desired **images**.


## Requirements

We have run the experiments on Linux. The versions are given in brackets. The following packages are used in the implementation:
* [PyTorch (1.4.0)](https://pytorch.org/get-started/locally/)
* [NumPy (1.17.3)](https://numpy.org/)
* [scikit-learn (0.22.1)](https://scikit-learn.org/stable/)
* [matplotlib (3.1.2)](https://matplotlib.org/)
* [tqdm (4.39.0)](https://github.com/tqdm/tqdm)

You can use [`pip`](https://pip.pypa.io/en/stable/) or [`conda`](https://docs.conda.io/en/latest/) to install them. We heavily used GPU in our experiments and recommend to use CUDA version of *PyTorch*.

## Contents

All the experiments can be found in the underlying notebooks:

| Notebook      | Description |
|-----------|------------|
|[notebooks/toy_experiment.ipynb](https://github.com/Daniil-Selikhanovych/neural-ot/blob/master/notebooks/toy_experiment.ipynb) | **Toy Experiment:** Transform Standard Multivariate Normal into multimodal mixture of Multivariate Normals.|
|[notebooks/domain_adaptation.ipynb](https://github.com/Daniil-Selikhanovych/neural-ot/blob/master/notebooks/domain_adaptation.ipynb) | **Domain Adaptation:** Turn the digits from [MNIST](http://yann.lecun.com/exdb/mnist/) into the [USPS](https://web.stanford.edu/~hastie/StatLearnSparsity_files/DATA/zipcode.html)-like digits and *vice versa*.  
|[notebooks/generative_modeling.ipynb](https://github.com/Daniil-Selikhanovych/neural-ot/blob/master/notebooks/generative_modeling.ipynb)| **Generative Modeling**: Transform a Multivariate Normal sample to the digit image. |
|[gaussian_learning_neural_ot/Gaussian_learning_OT.ipynb](https://github.com/Daniil-Selikhanovych/neural-ot/blob/master/gaussian_learning_neural_ot/Gaussian_learning_OT.ipynb)| **Additional Experiments**: Extra experiments with the Multivariate Normal Distribution and GIF generation.

For convenience, we have also implemented a framework and located it correspondingly in [neural_ot/](https://github.com/Daniil-Selikhanovych/neural-ot/tree/master/neural_ot) and [gaussian_learning_neural_ot/api/](https://github.com/Daniil-Selikhanovych/neural-ot/tree/master/gaussian_learning_neural_ot/api).

## Our team

At the moment we are *Skoltech DS MSc, 2019-2021* students.
* Artemenkov Aleksandr 
* Borzilov Alexander 
* Goncharov Mikhail
* Kornilova Anastasiia 
* Selikhanovych Daniil
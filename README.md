 1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miladnavi/few-shot-learning/blob/master/Google_Colab_Script.ipynb) - Label Preserving Transformations
2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miladnavi/few-shot-learning/blob/master/Bayesian_Google_Colab_Script.ipynb) - Bayesian Approach

3. [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)

# few-shot-learning
This repository consists of the code of Milad Navidizadeh's [Bachelor Thesis
](https://github.com/miladnavi/thesis/tree/master)in in fulfillment of requirements for degree
Bachelor of Science (B.Sc.).



## Component
- [PyTorch](https://pytorch.org/)
- [Augmentor](https://github.com/mdbloice/Augmentor)
- [Bayesian Approach](https://github.com/toantm/pytorch-bda)

## Installation
```python
pip install requirements.txt
```

## Usage

This project consists of following parts:

- Data Augmentation
    - Label Preserving Transformations
        - Image Translations
        - Elastic Distortions
        - Stroke Warping
        - Ensemble Learning & Label Preserving Transformations
        - Color Randomization & Ensemble Learning
    - Bayesian Approach
        - Random Erasing & Bayesian Approach

All above mentioned parts are available for following datasets:

- MNIST
- Fashion-MNIST
- Cifar-10

To run the code click on `Open in Colab` badges on the top of this page. You can run the code in
Google Colab. You can in Google Colab set the runtime environment on GPU, then the training will run on
GPU automatically. Otherwise copy the commands from the Jupyter notebook files ([Label Preserving
Transformations](https://github.com/miladnavi/few-shot-learning/blob/master/Google_Colab_Script.ipynb)
and 
[Bayesian
Approach](https://github.com/miladnavi/few-shot-learning/blob/master/Bayesian_Google_Colab_Script.ipynb))
into terminal respectively.

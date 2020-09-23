<p align="center">
<h3 align="center">Fast Image Processing with Fully-Convolutional Networks</h3>

<p align="center">
PyTorch implementation
<br />
<br />
<a href="https://github.com/nrupatunga/Fast-Image-Filters/issues">Report Bug</a>
·
<a href="https://github.com/nrupatunga/Fast-Image-Filters/issues">Request Feature</a>
</p>
</p> 

<!-- TABLE OF CONTENTS -->
## Table of Contents
* [About the Project](#about-the-project)
	- [Description](#description)
* [App Demo](#app-demo)
* [Getting Started](#getting-started)
	- [Code Setup](#code-setup)
	- [Data Download and Preparation](#data-download)
	- [Training](#training)
	- [Testing](#testing)

<!-- ABOUT THE PROJECT -->
## About The Project

This project is an extension to the previous project
[here](https://github.com/nrupatunga/pytorch-deaf) on edge aware
filters.

The link to paper is [here](https://arxiv.org/abs/1709.00643)

### Description

This paper shows effectiveness of the following in implementing image
processing filters

- Fully Convolutional architecture with dilated filters, that leads to
better receptive field for image filters and very smaller models
- Neat trick on Identity weight initialization
- Simple Mean Square Error loss

<!-- APP DEMO-->
## App Demo
|Image Filters|
|------------------------|
|![](https://github.com/nrupatunga/Fast-Image-Filters/blob/master/src/run/demo/output.gif)|

<!-- GETTING STARTED -->
## Getting Started

#### Code Setup
```
# Clone the repository
$ git clone https://github.com/nrupatunga/Fast-Image-Filters.git

# install all the required repositories
$ cd Fast-Image-Filters
$ pip install -r requirements.txt

# Add current directory to environment
$ cd src
$ source settings.sh
```

#### Data Download and Preparation

Since the author has not released the dataset and dataset preparation
script, I couldn't share the same here. Please mail cqf@ust.hk for the
scripts and data. Feel free to message me for any help you need.

#### Training
```
$ cd Fast-Image-Filters/src/run/

# Modify the data_dir variable in train.py

# run to train
$ python train.py
```

#### Testing
```
$ cd Fast-Image-Filters/src/run

# Change the model paths in app.py

$ python app.py
```

## Contact

Email: nrupatunga.tunga@gmail.com

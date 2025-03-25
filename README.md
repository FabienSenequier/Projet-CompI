# Score-Based Data Assimilation

Group 2 - Topic 2: NGUYEN Nhan - SENEQUIER Fabien - ARTHAUD Lilian

## Acknowledgement
This project was developed as part of the project of Computational Imaging course led and supervised by Ronan Fablet, François Rousseau and Mai Quyen Pham at IMT Atlantique - Bretagne-Pays de la Loire.

This repository contains an implementation of a score-based denoising model using reverse-time Stochastic Differential Equations (SDEs). The approach leverages a neural network to estimate the score function, which is then used to denoise images corrupted by Gaussian noise. 

Firsly, we try to do the implementation inspired by the work presented in the [Score-Based Data Assimilation](https://github.com/francois-rozet/sda/tree/master?tab=readme-ov-file) repository and the accompanying paper. Simultaneously, we develop our own denoising algorithm based on proposed methode and then apply it to one example image (only the script in [projet.ipynb](./projet.ipynb)).

## Table of Contents

1. [Introduction](#introduction)
2. [Score-Based Data Assimilation](#score-based-data-assimilation)
3. [Denoising Algorithm](#denoising-algorithm)

## Introduction

Score-based generative models have emerged as a powerful tool for tasks involving data denoising and generation. This implementation focuses on image denoising using a score network trained to estimate the gradient of the log-density of the data distribution. The denoising process is modeled as a reverse-time SDE, where the score network guides the removal of noise from the images.

## Score-Based Data Assimilation - Proposed Method by François ROZET

**Note**: For this project, the majority of the code is written in Python. Moreoever, we worked mainly on **Jupyter Notebook** and **Google Colab**. However, due to the limitted resources for the standard user of Google Colab, we manage to **use only the CPU** (the runtime type). One more thing to remember is that we work only on the [kolmogorov](./experiments/kolmogorov) directory because of time constraint of this project.
 
**Slide**: You can also find the final presentation of this project [here](./T2 _Gr2_ScoreBasedDataAssimilation.pdf). 

###  Organization
The [sda](./sda) directory contains the implementations of the [dynamical systems](./sda/sda/mcs.py), the [neural networks](./sda/sda/nn.py), the [score models](./sda/sda/score.py) and [various helpers](./sda/sda/utils.py).

The [kolmogorov](./experiments/kolmogorov) directory containes:

1. [generate.py](./experiments/kolmogorov/generate.py): training data generation.
2. [train.py](./experiments/kolmogorov/train.py): score model training.
3. [figures.ipynb](./experiments/kolmogorov/figures.ipynb): **main visualization** of inference from trained model.
4. [figures_bis.ipynb](./experiments/kolmogorov/figures_bis.ipynb): secondary visualization or new settings testing.
5. [sandwich.ipynb](./experiments/kolmogorov/sandwich.ipynb): vorticity fields and animations plotting. But since we use Google Colab which only allows to run the .py file within the notebook so we use this file to **generate data** and to **train model** too.

The [results](./results) directory containes all the simulated results and visualizations.

## Denoising Algorithm - Proposed Method by our team

### 1. Score Network Training 

#### Algorithm

You can find the training algorithm [here](./Training_algo.png).

1. **Data Preparation**: The training data consists of clean images from the CIFAR-10 dataset.
2. **Noise Addition**: Gaussian noise is added to the clean images to create noisy images.
3. **Score Estimation**: The score network is trained to estimate the score (gradient of the log-density) of the noisy images.
4. **Loss Function**: The training objective is to minimize the Mean Squared Error (MSE) between the predicted score and the true score.

#### Implementation

- **Model Architecture**: The score network is a simple convolutional neural network (CNN) with three convolutional layers.
- **Optimizer**: Adam optimizer is used with a learning rate of 0.0001.
- **Training Duration**: The network is trained for 200 epochs.

### 2. Denoising Algorithm

#### Algorithm

1. **Initialization**: Start with a noisy image.
2. **Iterative Denoising**: Use Langevin dynamics to iteratively update the image by moving it in the direction of the estimated score while adding controlled noise.
3. **Step Size**: The step size controls the magnitude of updates in each iteration.
4. **Intermediate Steps**: Optionally, store intermediate denoised images for visualization.

#### Implementation

- **Step Size**: A small step size (e.g., 0.0001) is used for stable updates.
- **Number of Steps**: The denoising process runs for a fixed number of steps (e.g., 100).

### 3. Usage

#### 1. Clone the Repository:
   ```bash
   git clone https://github.com/FabienSenequier/Projet-CompI.git
   cd Projet-CompI
   ```

#### 2. Open the Jupyter Notebook:

Navigate to the projet.ipynb file and open it using Jupyter Notebook or any compatible environment.
Run the notebook to train the score network and perform denoising on sample images. The results will be displayed within the notebook.
View the Presentation:

#### 3. Get more explanation:
Open the *slides.ppt* file for a detailed explanation of the project, including visual aids and additional context.

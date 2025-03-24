# Score-Based Denoising with Reverse-Time SDEs

NGUYEN Nhan - SENEQUIER Fabien - ARTHAUD Lilian

This repository contains an implementation of a score-based denoising model using reverse-time Stochastic Differential Equations (SDEs). The approach leverages a neural network to estimate the score function, which is then used to denoise images corrupted by Gaussian noise. This implementation is inspired by the work presented in the [Score-Based Data Assimilation](https://github.com/francois-rozet/sda/tree/master?tab=readme-ov-file) repository and the accompanying paper.

## Table of Contents

1. [Introduction](#introduction)
2. [Score Network Training](#score-network-training)
3. [Denoising Algorithm](#denoising-algorithm)
4. [Usage](#usage)

## Introduction

Score-based generative models have emerged as a powerful tool for tasks involving data denoising and generation. This implementation focuses on image denoising using a score network trained to estimate the gradient of the log-density of the data distribution. The denoising process is modeled as a reverse-time SDE, where the score network guides the removal of noise from the images.

## Score Network Training

### Algorithm

1. **Data Preparation**: The training data consists of clean images from the CIFAR-10 dataset.
2. **Noise Addition**: Gaussian noise is added to the clean images to create noisy images.
3. **Score Estimation**: The score network is trained to estimate the score (gradient of the log-density) of the noisy images.
4. **Loss Function**: The training objective is to minimize the Mean Squared Error (MSE) between the predicted score and the true score.

### Implementation

- **Model Architecture**: The score network is a simple convolutional neural network (CNN) with three convolutional layers.
- **Optimizer**: Adam optimizer is used with a learning rate of 0.0001.
- **Training Duration**: The network is trained for 200 epochs.

## Denoising Algorithm

### Algorithm

1. **Initialization**: Start with a noisy image.
2. **Iterative Denoising**: Use Langevin dynamics to iteratively update the image by moving it in the direction of the estimated score while adding controlled noise.
3. **Step Size**: The step size controls the magnitude of updates in each iteration.
4. **Intermediate Steps**: Optionally, store intermediate denoised images for visualization.

### Implementation

- **Step Size**: A small step size (e.g., 0.0001) is used for stable updates.
- **Number of Steps**: The denoising process runs for a fixed number of steps (e.g., 100).

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/FabienSenequier/Projet-CompI.git
   cd Projet-CompI

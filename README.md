WGAN_for_caustics_SINP

This repository contains the implementation of A WGAN GP model designed to reproduce phonon distributions from a G4CMP simulation of 0.1 eV energy deposition in Calcium Tungstate substrate. The input data is 2 Dimensional with shape (2.1 x 10^8, 4). The columns represent the 4 features X, Y, Energy and Time while the rows represent individual phonons. Codes used for this model along iwth their purposes are listed below. 

Input Processing: 

1. Separation of signal and noise : The input is highly noisy in X-Y dimensions, with the signal like pattern being hard to notice in small batch sizes. So we split the data into two sets, signal  (5.1x10^7 phonons)  and noise (1.5x10^8 phonons). code : seperation.py --> This code also save the min/max ranges of the original dataset, so that these values can be accessed later without loading the training set. 

2. Regularization and Quantile Transformation: Normalization to [0,1] for X and Y, Log transformation then normalization for E and T. This process is done to avoid any bias developed by the model to particular features. code : Input_processing.py --> Although this code saves the transformed dataset, this is not really needed since the transformaton is part of the WGAN training code. The main purpose of this is to save the Quantile transform functions, so that these can be accessed later without loading the original data set and doing the heavy computation again. This helps significantly reduce time during generation. QTs and range lists for noise and sigal are saved seperately. 

Training: 

We train two seperate WGAN models for the seperated sets. The underlying structure of these models are not different by any means. 
code : WGAN_noise.py, WGAN_signal.py


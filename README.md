WGAN_for_caustics_SINP

This repository contains the implementation of a WGAN GP model designed to reproduce the phonon distributions from a G4CMP simulation of 0.1 eV energy deposition in Calcium Tungstate substrate. The input data is 2 Dimensional with shape (2.1 x 10^8, 4). The columns represent the 4 features X, Y, Energy and Time while the rows represent individual phonons. Codes used for this model along with their purposes are listed below. 

Environment activation: source ~/Abhikamya/WGAN7/ml_v2/bin/activate

1.seperation.py

Separation of signal and noise : The input is highly noisy in X-Y dimensions, with the signal like pattern being hard to notice in small batch sizes. So we split the data into two sets, signal  (5.1x10^7 phonons)  and noise (1.5x10^8 phonons).This code also save the min/max ranges of the original dataset, so that these values can be accessed later without loading the training set. 


2.Input_processing.py 

Regularization and Quantile Transformation: Normalization to [0,1] for X and Y, Log transformation then normalization for E and T, then Quantile transformation of all 4 features. Although this code saves the transformed dataset, this is not really needed since the transformaton is part of the WGAN training code. The main purpose of this is to save the Quantile transform functions, so that these can be accessed later without loading the original data set and doing the heavy computation again. This helps significantly reduce time during generation. QTs and range lists for noise and sigal are saved seperately. 

3.WGAN_noise.py and  WGAN_signal.py
   
We train two seperate but identical WGAN models for the noise and signal sets. 

Algorithm used in the training is similar to,

<img width="1016" height="539" alt="image" src="https://github.com/user-attachments/assets/d4f9242d-cc88-4bee-9c9e-3070e511143b" />

with the hyper parameters changed according to the needs of our dataset. The training efficiency is improved by adding decaying gaussian noise to generated and real data. We have also implemented a warmup for the learning rates in both critic and generator. 

The models can be run on GPU or CPU but GPU is preferred due to its higher efficiency. The batch size is limited by memory, so you should not go higher than 128x128x4 in GPU and 128x128x32 in CPU ( assuming GPU ram of 16 GB and CPU ram of 100 GB). For the losses to be saved into memory, the code will need to finish running all given epochs. If you end up stopping the code in between these values are printed out on the terminal and can be recovered from there. Retraining from a previous trained model is also possible here. The model will load whichever version was saved last and start training it till the given epoch. It should be noted that if you want to start training from scratch on an already trained model, you should delete the checkpoints file, because if it exists and has saved check points, the model will only start training from there. 


4.generation.py 


The number of particles to be generated is taken as an input from the terminal. The code will find the number of noise and signal phonons to be made using a given ratio. It generates both noise and signal, does the inverse transformation, combines them and cleans up 'leakage'. These processes are sped up by accessing the previously saved ranges and QTs instead of having to load the original dataset. For generation of 10^7 particles, this code takes up about 4 minutes. Due to memory constraints only this many can be produced at a time. 




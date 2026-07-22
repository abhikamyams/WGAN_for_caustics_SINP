# SET CORE LIMITS FOR NUMPY AND PANDAS BEFORE IMPORTING THEM
import os
num_cores_str="20"
os.environ["OMP_NUM_THREADS"] = num_cores_str
os.environ["OPENBLAS_NUM_THREADS"] =num_cores_str
os.environ["MKL_NUM_THREADS"] = num_cores_str
os.environ["VECLIB_MAXIMUM_THREADS"] =num_cores_str 
os.environ["NUMEXPR_NUM_THREADS"] = num_cores_str

# SET GPU/CPU AND CORE LIMIT FOR TENSORFLOW
import tensorflow as tf

device='gpu'
if device =='cpu':
    tf.config.set_visible_devices([], 'GPU')

gpus = tf.config.get_visible_devices('GPU')

if gpus:
    print(f"GPUs available: {gpus}")
    print('Running on GPU')
else:
    print("No GPU found. Running on CPU.")
    num_cores=20
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)

# IMPORT RELEVANT LIBRARIES
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, Activation, Reshape, LayerNormalization, BatchNormalization
from tensorflow.keras.layers import Input, Dropout, Concatenate, Dense, LeakyReLU, Flatten
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow import convert_to_tensor
tf.experimental.numpy.experimental_enable_numpy_behavior()
AUTOTUNE = tf.data.experimental.AUTOTUNE
from sklearn.preprocessing import QuantileTransformer 
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import LeakyReLU
rng = np.random.default_rng()


# SET COLOR SCALE : if a pixel doesnt have a hit it will be black, this way we can distinguish between low no of hits and zero hits
cmap = plt.cm.viridis.copy()
cmap.set_under('black')  
plt.figure(figsize=(6,6))



# SETT HYPERPARAMETERS
MODEL_NAME = 'WCGAN'
features=4
sigma, decay = 1e-2, 0.9998 
BATCH_SIZE = 128*128*4
NOISE_DIM = 100
LAMBDA = 1e-4
EPOCHs = 60
CURRENT_EPOCH = 1 
SAVE_EVERY_N_EPOCH = 5 
add_losses=False
N_CRITIC = 3
count=0
warmup_steps=1000
learning_rate_generator = 1e-05
learning_rate_discriminator = 1e-05




# QUANTILE TRANSFORMATION FOR X,Y,E AND T
qtx = QuantileTransformer(output_distribution='normal', n_quantiles=1000000,subsample=20000000)
qty = QuantileTransformer(output_distribution='normal', n_quantiles=1000000,subsample=20000000)
qte = QuantileTransformer(output_distribution='normal', n_quantiles=1000000,subsample=20000000)
qtt = QuantileTransformer(output_distribution='normal', n_quantiles=1000000,subsample=20000000)




# LOAD TRAINING DATA
def load_real_samples():
    X=np.load('/home/ubuntu/Abhikamya/noise_removal/downconversion/signal_1.npy',allow_pickle=True)
	
    rng.shuffle(X, axis=0)
	#Only take 50% of the set for training due to time constraints
    X=X[:25711330,:]
    X=X.astype(np.float32)

	#Define the max/mins of the dataset for regularization
    maxx=np.max(X[:,0])
    minx=np.min(X[:,0])

    maxy=np.max(X[:,1]) 
    miny=np.min(X[:,1])

    maxe=np.max(X[:,2]) 
    mine=np.min(X[:,2]) 


    mint=np.min(X[:,3])
    maxt=np.max(X[:,3])
    
    logmint=np.min(np.log(X[:,3]))
    logmaxt=np.max(np.log(X[:,3]))

    logmine=np.min(np.log(X[:,2]))
    logmaxe=np.max(np.log(X[:,2]))
    return X,maxx,maxy,logmaxt,minx,miny,logmint,mint,maxt,mine,maxe,logmine,logmaxe


#Load the training set as a numpy array named train_data_bt of dimension ((2.5 x 10^ 7),4) 
train_data_bt,maxx,maxy,logmaxt,minx,miny,logmint,mint,maxt,mine,maxe,logmine,logmaxe = load_real_samples()

print('Total Number of Particles in dataset :',len(train_data_bt))





# NO OF BATCHES IN ONE EPOCH
n_batches=int(len(train_data_bt)/BATCH_SIZE)
print('Total Number of batches per epoch :', n_batches)



# TRANSFORMATION AND INVERSE TRANSFORMATION FUNCTIONS : 
# x: linear transformation --> quantile transformation to normal distribution 
# y: linear transformation --> quantile transformation to normal distribution 
# energy: log --> linear transformation --> quantile transformation to normal distribution 
# time: log --> linear transformation --> quantile transformation to normal distribution 

def transformation(Y):
	#Here a copy of Y is used because otherwise it seems to change the original dataset instead of just saving the transformed one as a new dataset. 
    X=Y.copy()

	#Regularization
    X[:,0]=(X[:,0] - minx)/(maxx-minx)
    X[:,1]=(X[:,1] - miny)/(maxy-miny)
    X[:,3]=(np.log(X[:,3])-logmint)/(logmaxt-logmint)
    X[:,2]=(np.log(X[:,2])-logmine)/(logmaxe-logmine)

	#Quantile Transformation
    X[:,0]=qtx.fit_transform(X[:,0].reshape(-1,1)).reshape(np.shape(X)[0])
    X[:,1]=qty.fit_transform(X[:,1].reshape(-1,1)).reshape(np.shape(X)[0])
    X[:,3]=qtt.fit_transform(X[:,3].reshape(-1,1)).reshape(np.shape(X)[0])
    X[:,2]=qte.fit_transform(X[:,2].reshape(-1,1)).reshape(np.shape(X)[0])

	#Convert to a tensor because the network accepts tensors as inputs
    X=convert_to_tensor(X)
    X = X.astype('float32')
    return X

def inverse_transformation(Y):
    X=Y.copy()

	#Inverse Transformation
    X[:,0]=qtx.inverse_transform(X[:,0].reshape(-1,1)).reshape(X.shape[0])
    X[:,1]=qty.inverse_transform(X[:,1].reshape(-1,1)).reshape(X.shape[0])
    X[:,3]=qtt.inverse_transform(X[:,3].reshape(-1,1)).reshape(X.shape[0])
    X[:,2]=qte.inverse_transform(X[:,2].reshape(-1,1)).reshape(X.shape[0])

    #Inverse Regularization
    X[:,3] = np.exp(X[:,3]*(logmaxt-logmint) + logmint)
    X[:,2] = np.exp(X[:,2]*(logmaxe-logmine) + logmine)
    X[:,0] = X[:,0]*(maxx-minx) + minx
    X[:,1] = X[:,1]*(maxy-miny) + miny
    return X



#sanity check to see if the inverse transformed data will be the same as the original dataset
#load the first 200 phonons
sampleset_real=train_data_bt[0:200,:]

#transform the whole data set and saving it as train_data_at
train_data_at=transformation(train_data_bt)


#inverse transformation of the first 200 phonons of train_data_at
sampleset_inversed=inverse_transformation(train_data_at.numpy()[0:200,:])


#calculate the relative difference between the first 200 from FullSim vs the First 200 after transformation and inverse transformation
diff=(sampleset_real-sampleset_inversed)/sampleset_real

print("/difference between real and inversed",np.sum(diff**2))








# KL DIVERGENCE CALCULATION

def get_kld(real: np.ndarray,
            fake: np.ndarray,
            bins: int = 12,
            epsilon: float = 1e-12):
    """
    Compute the KL-divergence between the distributions of real and fake samples.

    :param real: Tensor of shape (n_samples, n_features)
    :param fake: Tensor of shape (n_samples, n_features)
    :param bins: Number of bins per feature dimension (int) or sequence of ints
    :param epsilon: Small value to avoid log(0)
    :return: Scalar tensor containing the KL-divergence
    """

    # Move data to CPU numpy
    real_np = real
    fake_np = fake

    n_features = real_np.shape[1]
    print("n_features", n_features)
    # Build bin edges per dimension from real data
    if isinstance(bins, int):
        bin_counts = [bins] * n_features
    else:
        assert len(bins) == n_features, "bins must be int or sequence matching features"
        bin_counts = bins

    bin_edges = [np.linspace(real_np[:, i].min(), real_np[:, i].max(), b + 1)
                 for i, b in enumerate(bin_counts)]

    # Compute multidimensional histograms
    real_hist, _ = np.histogramdd(real_np, bins=bin_edges, density=True)
    fake_hist, _ = np.histogramdd(fake_np, bins=bin_edges, density=True)

    # Flatten and normalize to form probability mass functions
    real_p = real_hist.flatten()
    fake_p = fake_hist.flatten()
    real_p = real_p / (real_p.sum() + epsilon)
    fake_p = fake_p / (fake_p.sum() + epsilon)

    # Add epsilon to avoid zeros
    real_p = real_p + epsilon
    fake_p = fake_p + epsilon

    # Convert to torch tensors
    # real_t = torch.from_numpy(real_p).float()
    # fake_t = torch.from_numpy(fake_p).float()

    # Compute KL divergence: sum P_real * log(P_real / P_fake)
    kl = np.sum(real_p * (np.log(real_p) - np.log(fake_p)))
    return kl


# SLICING BASED ON BATCH SIZE
train_data_sliced=tf.data.Dataset.from_tensor_slices(train_data_at).batch(BATCH_SIZE,drop_remainder=True)


# SAVING A SAMPLE PLOT OF ONE BATCH
def save_plot2(X):
    #load back to numpy
    X=X.numpy()
	
	#inverse transformation
    X=inverse_transformation(X)



    plt.figure(figsize=(30,25))

    plt.subplot(3,3,1,title='XY',xlabel='X in m', ylabel='Y in m')
    hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,range=[[minx,maxx],[miny,maxy]],vmin=0.1)
    cbar1 = plt.colorbar()
    cbar1.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,2,title='XT full',xlabel='X in m', ylabel='Time in ns')
    hist=plt.hist2d(X[:,0],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[minx,maxx],[mint,maxt]])
    plt.yscale('log')
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,3,title='XT zoom',xlabel='X in m', ylabel='Time in ns')
    hist=plt.hist2d(X[:,0],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[minx,maxx],[0,5e+4]])
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)


    plt.subplot(3,3,4,title='YT full',xlabel='Y in m', ylabel='Time in ns')
    hist=plt.hist2d(X[:,1],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[miny,maxy],[mint,maxt]])
    plt.yscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,5,title='YT zoomed',xlabel='Y in m', ylabel='Time in ns')
    hist=plt.hist2d(X[:,1],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[miny,maxy],[0,5e+4]])
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)
    plt.suptitle('Signal for Initial Position X = 0')

    plt.subplot(3,3,6,title='ET full',xlabel='Energy in eV ', ylabel='Time in ns')
    hist=plt.hist2d(X[:,2],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[mine,maxe],[mint,maxt]])
    plt.yscale('log')
    plt.xscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,7,title='ET zoomed',xlabel='Energy in eV ', ylabel='Time in ns')
    hist=plt.hist2d(X[:,2],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[mine,maxe],[0,5e+4]])
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,8,title='XE',xlabel='X in m', ylabel='Energy in eV ')
    hist=plt.hist2d(X[:,0],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[minx,maxx],[mine,maxe]])
    plt.yscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,9,title='YE',xlabel='Y in m', ylabel='Energy in eV ')
    hist=plt.hist2d(X[:,1],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[miny,maxy],[mine,maxe]])
    plt.yscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)



    plt.suptitle('sample of one batch')


    filename = '128*128*4.png'
    plt.savefig(filename)
    plt.close()

sample_image=next(iter(train_data_sliced))
save_plot2(sample_image)







                                              #GENERATOR LAYERS : 8 Dense layers ,Neurons per layer : 100-->512-->512-->1024x3-->512-->4


def CGAN_generator(input_z_shape=NOISE_DIM):
    '''
        DCGAN like generator architecture
    '''
    input_z_layer = Input(shape=(input_z_shape,))
#1
    x = Dense(512, use_bias=False)(input_z_layer)
    x = BatchNormalization(scale=True,center=True)(x)
    x = ReLU()(x)
#2
    x = Dense(512, use_bias=False)(x)
    x = BatchNormalization(scale=True,center=True)(x)
    x = ReLU()(x)
#3

    x = Dense(1024, use_bias=False)(x)
    x = BatchNormalization(scale=True,center=True)(x)
    x = ReLU()(x)
#4
    x = Dense(1024, use_bias=False)(x)
    x = BatchNormalization(scale=True,center=True)(x)
    x = ReLU()(x)
#5
    x = Dense(1024, use_bias=False)(x)
    x = BatchNormalization(scale=True,center=True)(x)
    x = ReLU()(x)
#6
    x = Dense(1024, use_bias=False)(x)
    x = BatchNormalization(scale=True,center=True)(x)
    x = ReLU()(x)
#7
    x = Dense(512, use_bias=False)(x)
    x = BatchNormalization(scale=True,center=True)(x)
    x = ReLU()(x)
#8
    output = Dense(features, use_bias=True)(x)
 

    model = Model(inputs=input_z_layer, outputs=output)
    return model

                                                     #DISCRIMINATOR LAYERS : 8 Dense layers ,Neurons per layer : 4-->512x6-->1
def CGAN_discriminator(input_x_shape=(features,)):
    '''
        DCGAN like discriminator architecture
    '''
    input_x_layer = Input(shape=input_x_shape)
#1
    x = Dense(512, use_bias=True)(input_x_layer)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
#2
    x = Dense(512, use_bias=True)(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
#3
    x = Dense(512, use_bias=True)(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
#4
    x = Dense(512, use_bias=True)(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
#5

    x = Dense(512, use_bias=True)(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
#5
    x = Dense(512, use_bias=True)(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
#6
    x = Dense(512, use_bias=True)(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
#7
    output = Dense(1, use_bias=False)(x)


    model = Model(inputs=input_x_layer, outputs=output)
    return model





generator = CGAN_generator()





generator.summary()


discriminator = CGAN_discriminator()



discriminator.summary()




#set optimizer hyper parameters
D_optimizer = Adam(learning_rate=1e-05,     beta_1=0,
    beta_2=0.9)
G_optimizer = Adam(learning_rate=1e-05,     beta_1=0,
    beta_2=0.9)


def set_learning_rate(lr_g,lr_d):
    '''
        Set new learning rate to optimizers
    '''
    D_optimizer.learning_rate.assign(lr_d)
    G_optimizer.learning_rate.assign(lr_g)


#warm up function to increase learning rate gradually 
def learning_rate(lr,step):
    step=min(step,warmup_steps)
    factor=0.5*(1-np.cos(step/warmup_steps*np.pi))
    return lr*factor


#Set checkpoints to regularly save the model for retraining as well as generation.
checkpoint_path = os.path.join("checkpoints", "tensorflow", MODEL_NAME)
ckpt = tf.train.Checkpoint(generator=generator,discriminator=discriminator,G_optimizer=G_optimizer,D_optimizer=D_optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,max_to_keep=None)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    latest_epoch = int(ckpt_manager.latest_checkpoint.split('-')[1])
	#update the current epoch
    CURRENT_EPOCH = latest_epoch * SAVE_EVERY_N_EPOCH+1
	#update the count used for warmup
    count=latest_epoch*n_batches
	#update the sigma used for gaussian noise addition
    sigma=sigma*(decay**count)
	#add the new losses to the ones from previous training
    add_losses=True
    print ('Latest checkpoint of epoch {} restored!!'.format(CURRENT_EPOCH-1))

    




#Generate images using the current version of generator and plot 2D correlations
def generate_and_save_images(model, epoch, test_input,real,save=True):
    '''
        Generate images and plot it.
    '''
	#when we use model.predict, the output is already a numpy file so there is no need for .to_numpy()
    X = model.predict(test_input)
	#Inverse transform the generated data
    X=inverse_transformation(X)

	#kl divergence between 10^5 particles from FullSim Vs FastSim
    kl=get_kld(real,X[0:100000,:])

	#plotting
    plt.figure(figsize=(30,25))

	
    plt.subplot(3,3,1,title='XY',xlabel='X in m', ylabel='Y in m')
    hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,range=[[minx,maxx],[miny,maxy]],vmin=0.1)
    cbar1 = plt.colorbar()
    cbar1.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,2,title='XT full',xlabel='X in m', ylabel='Time in ns')
    hist=plt.hist2d(X[:,0],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[minx,maxx],[mint,maxt]])
    plt.yscale('log')
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,3,title='XT zoom',xlabel='X in m', ylabel='Time in ns')
    hist=plt.hist2d(X[:,0],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[minx,maxx],[0,5e+4]])
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)


    plt.subplot(3,3,4,title='YT full',xlabel='Y in m', ylabel='Time in ns')
    hist=plt.hist2d(X[:,1],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[miny,maxy],[mint,maxt]])
    plt.yscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,5,title='YT zoomed',xlabel='Y in m', ylabel='Time in ns')
    hist=plt.hist2d(X[:,1],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[miny,maxy],[0,5e+4]])
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)
    plt.suptitle('Signal for Initial Position X = 0')

    plt.subplot(3,3,6,title='ET full',xlabel='Energy in eV ', ylabel='Time in ns')
    hist=plt.hist2d(X[:,2],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[mine,maxe],[mint,maxt]])
    plt.yscale('log')
    plt.xscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,7,title='ET zoomed',xlabel='Energy in eV ', ylabel='Time in ns')
    hist=plt.hist2d(X[:,2],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[mine,maxe],[0,5e+4]])
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,8,title='XE',xlabel='X in m', ylabel='Energy in eV ')
    hist=plt.hist2d(X[:,0],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[minx,maxx],[mine,maxe]])
    plt.yscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,9,title='YE',xlabel='Y in m', ylabel='Energy in eV ')
    hist=plt.hist2d(X[:,1],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[miny,maxy],[mine,maxe]])
    plt.yscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)



    plt.suptitle(f'Histograms for caustics 1/10, after epoch {epoch}')

    filename = 'caustics_after_e%03d.png' % (epoch)
    plt.savefig(filename)
    plt.close()
    return kl

num_examples_to_generate = BATCH_SIZE


#only generate one batch at a time for memory efficiency 
num_examples_to_generate = BATCH_SIZE

# Create a random seed of dimension (num_examples_to_generate,NOISE_DIM).
#This seed is used every time for plotting as a reference to see how the generator transforms the same noise at different epochs. 
sample_noise = tf.random.normal([num_examples_to_generate, NOISE_DIM])


#generate and save one batch with the current generator ( maybe untrained or just loaded for retraining ) 
kl_sample=generate_and_save_images(generator, CURRENT_EPOCH-1, sample_noise,train_data_bt[0:100000,:], save=True)

#Kl divergence for one batch
print("kl_divergence_for_epoch0",kl_sample)



#Training step for discriminator
@tf.function
def WGAN_GP_train_d_step(real_image,sigma, batch_size=BATCH_SIZE):
    '''
        One discriminator training step

        Reference: https://www.tensorflow.org/tutorials/generative/dcgan
    '''

	#Create noise for Generator
    noise = tf.random.normal([batch_size, NOISE_DIM])

	#Epsilon to create a mixed set between tarining data and generated data. 
    epsilon = tf.random.uniform(shape=[batch_size, 1], minval=0, maxval=1)
    ###################################
    # Train D
    ###################################

	#add random noise to the real set
    real_noisy= real_image + sigma * tf.random.normal(tf.shape(real_image))

	#here gradient tape is what calculated all the needed gradients for optimization. 
	#d_tape used for the total critic losss gradient
    with tf.GradientTape(persistent=True) as d_tape:
		#gp_tape used for gradient penalty calculation. Since gradient penalty is a term inside critic loss, gp_tape is nested insdide d_tape.
        with tf.GradientTape() as gp_tape:
            fake_image = generator(noise, training=True)
			#add random noise to the fake set
            fake_noisy= fake_image + sigma * tf.random.normal(tf.shape(fake_image))
			#mixed set for GP
            fake_image_mixed = epsilon * tf.dtypes.cast(real_noisy, tf.float32) + ((1 - epsilon) * fake_noisy)
			#produce discriminator outputs for the mixed set
            fake_mixed_pred = discriminator(fake_image_mixed, training=True)

        # Compute gradient penalty
        grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))

		# Compute critic loss
        fake_pred = discriminator(fake_noisy, training=True)
        real_pred = discriminator(real_noisy, training=True)
        W_dist=tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
        D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + LAMBDA * gradient_penalty
	
    # Calculate the gradients for discriminator
    D_gradients = d_tape.gradient(D_loss,
                                            discriminator.trainable_variables)
    # Apply the gradients to the optimizer
    D_optimizer.apply_gradients(zip(D_gradients,
                                                discriminator.trainable_variables))
    return W_dist, LAMBDA*gradient_penalty




@tf.function
def WGAN_GP_train_g_step(real_image,sigma,batch_size=BATCH_SIZE):
    '''
        One generator training step

        Reference: https://www.tensorflow.org/tutorials/generative/dcgan
    '''
	#Create noise for Generator
    noise = tf.random.normal([batch_size, NOISE_DIM])
    ###################################
    # Train G
    ###################################
	#g_tape used for the total generator loss gradient
    with tf.GradientTape() as g_tape:
        fake_image = generator(noise, training=True)
        fake_noisy= fake_image + sigma * tf.random.normal(tf.shape(fake_image))
        fake_pred = discriminator(fake_noisy, training=True)
        G_loss = -tf.reduce_mean(fake_pred)
	
    # Calculate the gradients for G_loss
    G_gradients = g_tape.gradient(G_loss,
                                            generator.trainable_variables)
    # Apply the gradients to the optimizer
    G_optimizer.apply_gradients(zip(G_gradients,
                                                generator.trainable_variables))
    return G_loss



#create empty lists to log losses, gp and kld
D_lossl=[]
G_lossl=[]
Gradient_penalty=[]
kll=[]
for epoch in range(CURRENT_EPOCH, EPOCHs + 1):
    start = time.time()
    print('Start of epoch %d' % (epoch,))
	#calculate losses for critic(lde) and generator(lge) and gradient penalty(gpe) per step, then take the average over one epoch.  
    lde=0
    lge=0
    gpe=0
    batch_per_epo=n_batches
    for step, (image) in enumerate(train_data_sliced):
		#learning rate is updated every step
        lr_g = learning_rate(learning_rate_generator,count)
        lr_d = learning_rate(learning_rate_discriminator,count)
        set_learning_rate(lr_g,lr_d)
		
        # Train critic (discriminator) Ncritic times on the same batch
        for i in range(N_CRITIC):
            D,gp=WGAN_GP_train_d_step(image,tf.constant(sigma, dtype=tf.float32))
			#add critic loss and gradient penalty
            gpe=gpe+gp
            lde=lde+D
		#Train generator once per batch
        G=WGAN_GP_train_g_step(image,tf.constant(sigma, dtype=tf.float32))

        #add generator loss
        lge=lge+G
        
        if step % 10 == 0:
            print ('.', end='')

		#update sigma for noise addition
        sigma=sigma*decay

		#update count for warmup
        count=count+1

    # Use a consistent sample so that the progress of the model is clearly visible.
    kl=generate_and_save_images(generator, epoch, sample_noise,train_data_bt[0:100000,:], save=True)


	#SAVE_EVERY_N_EPOCH
    if epoch % SAVE_EVERY_N_EPOCH == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch,
                                                             ckpt_save_path))
	
    #append the average of losses and gradient penalty. 
    D_lossl.append(-lde/(n_batches*N_CRITIC))
    G_lossl.append(lge/n_batches)
    Gradient_penalty.append(gpe/(n_batches*N_CRITIC))
    kll.append(kl)
    print('>%d, %d/%d, wdist=%.7f, g=%.7f gp=%.7f kl=%.7f' %
				(epoch+1, step+1, batch_per_epo, -lde/(n_batches*N_CRITIC), lge/n_batches, gpe/(n_batches*N_CRITIC),kl))
    print ('Time taken for epoch {} is {} minutes\n'.format(epoch,
                                                      (time.time()-start)/60))



#save at the very last epoch, if not already saved
if EPOCHs % SAVE_EVERY_N_EPOCH != 0: 
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(EPOCHs,
                                                        ckpt_save_path))



#if a previous check point has been saved, reload the losses and append the new ones to that list
if add_losses==True:
    results=np.load('results.npy')
    [D_lossl_o,G_lossl_o,kll_o,Gradient_penalty_o] = results
    D_lossl=np.append(D_lossl_o,D_lossl)
    G_lossl=np.append(G_lossl_o,G_lossl)
    kll=np.append(kll_o,kll)
    Gradient_penalty=np.append(Gradient_penalty_o,Gradient_penalty)


#save the lossfunctions as results.npy
results= [D_lossl,G_lossl,kll,Gradient_penalty]
results=np.array(results)
np.save('results',results)
print('losses, kl divergence and gp saved')

plt.close("all")
plt.figure(figsize=(12,12))


#plot losses, gp and kld

epochs_n=[i+1 for i in range (len(D_lossl))]
plt.plot(epochs_n,D_lossl,label="W distance")
plt.plot(epochs_n,G_lossl,label="Generator loss")
plt.plot(epochs_n,Gradient_penalty,label="Gradient penalty")
plt.xlabel('epochs')
plt.legend()
plt.savefig('loss_functions_and_gp.png')
plt.close("all")


plt.plot(epochs_n,D_lossl,label="W distance")
plt.plot(epochs_n,G_lossl,label="Generator loss")
plt.legend()
plt.xlabel('epochs')
plt.savefig('loss_functions.png')
plt.close("all")

plt.plot(epochs_n,D_lossl,label="W distance")
plt.plot(epochs_n,G_lossl,label="Generator loss")
plt.xlabel('epochs')
plt.legend()
plt.savefig('loss_functions.png')
plt.close("all")

plt.plot(epochs_n,kll,label="KL_divergence")
plt.yscale('log')
plt.xlabel('epochs')
plt.title('KL divergence')
plt.savefig('kl_divergence.png')


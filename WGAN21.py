#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division


import os
num_cores_str="25"
os.environ["OMP_NUM_THREADS"] = num_cores_str
os.environ["OPENBLAS_NUM_THREADS"] =num_cores_str
os.environ["MKL_NUM_THREADS"] = num_cores_str
os.environ["VECLIB_MAXIMUM_THREADS"] =num_cores_str 
os.environ["NUMEXPR_NUM_THREADS"] = num_cores_str


import tensorflow as tf
print("Physical Devices:")
print(tf.config.list_physical_devices())
device='gpu'

if device =='cpu':
    tf.config.set_visible_devices([], 'GPU')

gpus = tf.config.get_visible_devices('GPU')

if gpus:
    print(f"GPUs available: {gpus}")
else:
    print("No GPU found. Running on CPU.")
    num_cores=25
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)


import time
import math
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output


from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, Activation, Reshape, LayerNormalization, BatchNormalization
from tensorflow.keras.layers import Input, Dropout, Concatenate, Dense, LeakyReLU, Flatten
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import plot_model
from tensorflow import convert_to_tensor
tf.experimental.numpy.experimental_enable_numpy_behavior()
AUTOTUNE = tf.data.experimental.AUTOTUNE
from sklearn.preprocessing import QuantileTransformer 
from tensorflow.keras.layers import GroupNormalization
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import LeakyReLU
# # Hyper parameters

rng = np.random.default_rng()
cmap = plt.cm.viridis.copy()
cmap.set_under('black')  
# In[2]:


MODEL_NAME = 'WCGAN'

features=2

sigma, decay = 1e-2, 0.9998 
BATCH_SIZE = 128*128*4
NOISE_DIM = 100
LAMBDA = 5e-4 # For gradient penalty

EPOCHs = 30
CURRENT_EPOCH = 1 # Epoch start from
SAVE_EVERY_N_EPOCH = 5 # Save checkpoint at every n epoch
add_losses=False
N_CRITIC = 3 # Train critic(discriminator) n times then train generator 1 time.


count=0
warmup_steps=1000
 


learning_rate_generator = 1e-05
learning_rate_discriminator = 1e-05



qtx = QuantileTransformer(output_distribution='normal', n_quantiles=100000,subsample=100000)
qty = QuantileTransformer(output_distribution='normal', n_quantiles=100000,subsample=100000)
qte = QuantileTransformer(output_distribution='normal', n_quantiles=100000,subsample=100000)
qtt = QuantileTransformer(output_distribution='normal', n_quantiles=100000,subsample=100000)
# In[4]:
plt.figure(figsize=(6,6))

def load_real_samples():
    X=np.load('/home/ubuntu/Abhikamya/noise_removal/small_input_100000.npy')
    rng.shuffle(X, axis=0)
    X = np.delete(X, 3,1)
    X = np.delete(X, 2,1)
    
    maxx=np.max(X[:,0])
    maxy=np.max(X[:,1]) 
    minx=np.min(X[:,0])
    miny=np.min(X[:,1]) 


    # print(np.shape(X))
    # plt.title('X,Y and Time distributions before quantile transformation')

    # plt.subplot(1,3,1)
    # plt.hist(X[:,0],bins=1000)
    # plt.subplot(1,3,2)
    # plt.hist(X[:,1],bins=1000)
    # plt.subplot(1,3,3)
    # plt.hist(X[:,3],bins=1000)
    # plt.tight_layout()
    # plt.savefig('Before_Quantile_transformation.png')

    # X[:,0] = qtx.fit_transform(X[:,0].reshape(-1,1)).reshape(np.shape(X)[0])
    # X[:,1] = qty.fit_transform(X[:,1].reshape(-1,1)).reshape(np.shape(X)[0])
    # X[:,2] = qte.fit_transform(np.log(X[:,2].reshape(-1,1))).reshape(np.shape(X)[0])
    # X[:,3] = qtt.fit_transform(np.log(X[:,3].reshape(-1,1))).reshape(np.shape(X)[0])
    # X=convert_to_tensor(X)
    # X = X.astype('float32')
    # print('training data shape',np.shape(X))
    return X,maxx,maxy,minx,miny


train_data_bt,maxx,maxy,minx,miny = load_real_samples()
print('Total Number of Particles in dataset :',len(train_data_bt))

n_batches=int(len(train_data_bt)/BATCH_SIZE)

print('Total Number of batches per epoch :', n_batches)
def transformation(Y):
    X=Y.copy()
    X[:,0] = (X[:,0] - minx)/(maxx-minx)
    X[:,1] = (X[:,1] - miny)/(maxy-miny)
    X[:,0] = qtx.fit_transform(X[:,0].reshape(-1,1)).reshape(np.shape(X)[0])
    X[:,1] = qty.fit_transform(X[:,1].reshape(-1,1)).reshape(np.shape(X)[0])
    # X[:,2] = qte.fit_transform(np.log(X[:,2].reshape(-1,1))).reshape(np.shape(X)[0])
    # X[:,3] = qtt.fit_transform(np.log(X[:,3].reshape(-1,1))).reshape(np.shape(X)[0])
    X=convert_to_tensor(X)
    X = X.astype('float32')
   
    return X

def inverse_transformation(X):
    # X=Y.copy()
    X[:,0]=qtx.inverse_transform(X[:,0].reshape(-1,1)).reshape(X.shape[0])
    X[:,1]=qty.inverse_transform(X[:,1].reshape(-1,1)).reshape(X.shape[0])
    X[:,0] = X[:,0]*(maxx-minx) + minx
    X[:,1] = X[:,1]*(maxy-miny) + miny
    return X


sampleset_real=train_data_bt[0:200,:]
# print("real1",sampleset_real)
train_data_at=transformation(train_data_bt)
# print(train_data_at)
# print("real2",sampleset_real)
# diff=train_data_at-train_data_bt
# print("differece",np.sum(diff))
sampleset_inversed=inverse_transformation(train_data_at.numpy()[0:200,:])

# print(sampleset_inversed)

diff=sampleset_real-sampleset_inversed

print("/difference between real and inversed",np.sum(diff**2))



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

# plt.close('all')
# plt.figure(figsize=(18,6))
# plt.title('X,Y and Time distributions after quantile transformation')
# X=train_data
# plt.subplot(1,3,1)
# plt.hist(X[:,0],bins=1000)
# plt.subplot(1,3,2)
# plt.hist(X[:,1],bins=1000)
# plt.subplot(1,3,3)
# plt.hist(X[:,3],bins=1000)
# plt.tight_layout()
# plt.savefig('After_Quantile_transformation.png')
# plt.close()


# def save_plot(X,real):
#     plt.figure(figsize=(6,6))
#     X=X.numpy()
#     X=inverse_transformation(X)
   
#     kl=get_kld(real,X[0:100000,:])
#     n=1
#     hist=plt.hist2d(X[:,0],X[:,1],bins=60,cmap=cmap,range=[[-0.02,0.02],[-0.02,0.02]],vmin=0.1)
#     filename = 'sample.png'
#     plt.savefig(filename)
#     plt.close()
#     return kl


# plt.title('Sample')

# kl_same=save_plot(train_data_at,inverse_transformation(train_data_at.numpy()[0:100000,:]))

# print('after inverse transform',train_data_bt[0:5,:])
# print("kl divergence for same set", f"{kl_same:.20f}")

# kl_bt_and_at=save_plot(train_data_at,train_data_bt[0:100000,:])

# print("kl divergence for before and after transformation", f"{kl_bt_and_at:.20f}")

train_data_sliced=tf.data.Dataset.from_tensor_slices(train_data_at).batch(BATCH_SIZE,drop_remainder=True)

# #train_data=tf.data.Dataset.load_real_samples().shuffle(1000).batch(BATCH_SIZE)

def save_plot2(X):
    plt.figure(figsize=(6,6))
    X=X.numpy()
    X=inverse_transformation(X)
    hist=plt.hist2d(X[:,0],X[:,1],bins=60,cmap=cmap,range=[[-0.02,0.02],[-0.02,0.02]],vmin=0.1)
    plt.xlabel('x in cms', fontsize=11)
    plt.ylabel('y in cms', fontsize=11)
    plt.title('sample of one batch', pad=15)
    cbar = plt.colorbar()
    cbar.set_label('Counts in each pixel', labelpad=15, fontsize=11)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.82, top=0.85)
    filename = '128*128*4.png'
    plt.savefig(filename)
    plt.close()

sample_image=next(iter(train_data_sliced))
save_plot2(sample_image)






# # In[7]:
                                                                    #GENERATOR LAYERS : 6

# # You could also try layer normalization instead of batch normalization

def CGAN_generator(input_z_shape=NOISE_DIM):
    '''
        DCGAN like generator architecture
    '''
    input_z_layer = Input(shape=(input_z_shape,))

    x = Dense(512, use_bias=False)(input_z_layer)
    x = BatchNormalization(scale=True,center=True)(x)
    x = ReLU()(x)

    x = Dense(1024, use_bias=False)(x)
    x = BatchNormalization(scale=True,center=True)(x)
    x = ReLU()(x)

    x = Dense(1024, use_bias=False)(x)
    x = BatchNormalization(scale=True,center=True)(x)
    x = ReLU()(x)

    x = Dense(1024, use_bias=False)(x)
    x = BatchNormalization(scale=True,center=True)(x)
    x = ReLU()(x)

    x = Dense(512, use_bias=False)(x)
    x = BatchNormalization(scale=True,center=True)(x)
    x = ReLU()(x)

    output = Dense(features, use_bias=True)(x)
 

    model = Model(inputs=input_z_layer, outputs=output)
    return model

                                                                    #DISCRIMINATOR LAYERS : 6
def CGAN_discriminator(input_x_shape=(features,)):
    '''
        DCGAN like discriminator architecture
    '''
    input_x_layer = Input(shape=input_x_shape)

    x = Dense(512, use_bias=True)(input_x_layer)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(512, use_bias=True)(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(512, use_bias=True)(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(512, use_bias=True)(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(512, use_bias=True)(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)

    output = Dense(1, use_bias=False)(x)


    model = Model(inputs=input_x_layer, outputs=output)
    return model





generator = CGAN_generator()





generator.summary()


discriminator = CGAN_discriminator()



discriminator.summary()




# # Optimizers 
D_optimizer = Adam(learning_rate=1e-05,     beta_1=0.5,
    beta_2=0.9)
G_optimizer = Adam(learning_rate=1e-05,     beta_1=0.5,
    beta_2=0.9)


def set_learning_rate(lr_g,lr_d):
    '''
        Set new learning rate to optimizers
    '''
    D_optimizer.learning_rate.assign(lr_d)
    G_optimizer.learning_rate.assign(lr_g)


# # # Setup Checkpoints

# # In[14]:


checkpoint_path = os.path.join("checkpoints", "tensorflow", MODEL_NAME)

ckpt = tf.train.Checkpoint(generator=generator,discriminator=discriminator,G_optimizer=G_optimizer,D_optimizer=D_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,max_to_keep=None)

# # # if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    latest_epoch = int(ckpt_manager.latest_checkpoint.split('-')[1])
    CURRENT_EPOCH = latest_epoch * SAVE_EVERY_N_EPOCH+1
    count=latest_epoch*n_batches
    sigma=sigma*(decay**count)
    add_losses=True
    print ('Latest checkpoint of epoch {} restored!!'.format(CURRENT_EPOCH-1))

    
# # In[15]:


def generate_and_save_images(model, epoch, test_input,real, figure_size=(12,12), save=True):
    '''
        Generate images and plot it.
    '''

    X = model.predict(test_input)
    X=inverse_transformation(X)

    kl=get_kld(real,X[0:100000,:])

    hist=plt.hist2d(X[:,0],X[:,1],bins=60,cmap=cmap,range=[[-0.02,0.02],[-0.02,0.02]], vmin = 0.1)
    plt.title(f'Histograms for caustics 1/10, after epoch {epoch}',pad=15)
    plt.xlabel('x in cms', fontsize=11)
    plt.ylabel('y in cms', fontsize=11)
    cbar = plt.colorbar()
    cbar.set_label('Counts in each pixel', labelpad=15, fontsize=11)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.82, top=0.85)
    filename = 'caustics_after_e%03d.png' % (epoch)
    plt.savefig(filename)
    plt.close()
    return kl


num_examples_to_generate = 128*128*4

# # We will reuse this seed overtime
sample_noise = tf.random.normal([num_examples_to_generate, NOISE_DIM])
kl_sample=generate_and_save_images(generator, CURRENT_EPOCH-1, sample_noise,train_data_bt[0:100000,:], figure_size=(12,12), save=True)
print("kl_divergence_for_epoch0",kl_sample)

# # # Define training step

# # In[18]:


def learning_rate(lr,step):
    step=min(step,warmup_steps)
    factor=0.5*(1-np.cos(step/warmup_steps*np.pi))
    return lr*factor
@tf.function
def WGAN_GP_train_d_step(real_image,sigma, batch_size=BATCH_SIZE):
    '''
        One discriminator training step

        Reference: https://www.tensorflow.org/tutorials/generative/dcgan
    '''

    noise = tf.random.normal([batch_size, NOISE_DIM])
    epsilon = tf.random.uniform(shape=[batch_size, 1], minval=0, maxval=1)
    ###################################
    # Train D
    ###################################
    real_noisy= real_image + sigma * tf.random.normal(tf.shape(real_image))
    with tf.GradientTape(persistent=True) as d_tape:
        with tf.GradientTape() as gp_tape:
            fake_image = generator(noise, training=True)
            fake_noisy= fake_image + sigma * tf.random.normal(tf.shape(fake_image))
            fake_image_mixed = epsilon * tf.dtypes.cast(real_noisy, tf.float32) + ((1 - epsilon) * fake_noisy)
            fake_mixed_pred = discriminator(fake_image_mixed, training=True)

        # Compute gradient penalty
        grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))


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
    # Write loss values to tensorboard
    # if step % 10 == 0:
    #     with file_writer.as_default():
    #         tf.summary.scalar('D_loss', tf.reduce_mean(D_loss), step=step)
    return W_dist, LAMBDA*gradient_penalty

@tf.function
def WGAN_GP_train_g_step(real_image,sigma,batch_size=BATCH_SIZE):
    '''
        One generator training step

        Reference: https://www.tensorflow.org/tutorials/generative/dcgan
    '''
    noise = tf.random.normal([batch_size, NOISE_DIM])
    ###################################
    # Train G
    ###################################
    with tf.GradientTape() as g_tape:
        fake_image = generator(noise, training=True)
        fake_noisy= fake_image + sigma * tf.random.normal(tf.shape(fake_image))
        fake_pred = discriminator(fake_noisy, training=True)
        G_loss = -tf.reduce_mean(fake_pred)
    # Calculate the gradients for generator
    G_gradients = g_tape.gradient(G_loss,
                                            generator.trainable_variables)
    # Apply the gradients to the optimizer
    G_optimizer.apply_gradients(zip(G_gradients,
                                                generator.trainable_variables))
    # Write loss values to tensorboard
    # if step % 10 == 0:
    #     with file_writer.as_default():
    #         tf.summary.scalar('G_loss', G_loss, step=step)
    return G_loss


# # # Start training

# # In[19]:

# trace = True
n_critic_count = 0

D_lossl=[]
G_lossl=[]
Gradient_penalty=[]
kll=[]
for epoch in range(CURRENT_EPOCH, EPOCHs + 1):
    start = time.time()
    print('Start of epoch %d' % (epoch,))
    # Using learning rate decay
    lde=0
    lge=0
    gpe=0
    batch_per_epo=n_batches
    for step, (image) in enumerate(train_data_sliced):
        lr_g = learning_rate(learning_rate_generator,count)
        lr_d = learning_rate(learning_rate_discriminator,count)
        set_learning_rate(lr_g,lr_d)
        # Train critic (discriminator)
        for i in range(N_CRITIC):
            D,gp=WGAN_GP_train_d_step(image,tf.constant(sigma, dtype=tf.float32))
            # Train generator
            gpe=gpe+gp
            lde=lde+D
        G=WGAN_GP_train_g_step(image,tf.constant(sigma, dtype=tf.float32))

        
        lge=lge+G
        
        if step % 10 == 0:
            print ('.', end='')
        sigma=sigma*decay
        count=count+1

    # Using a consistent sample so that the progress of the model is clearly visible.
    # generate_and_save_images(generator, epoch, sample_noise, figure_size=(12,12), save=True)

    kl=generate_and_save_images(generator, epoch, sample_noise,train_data_bt[0:100000,:], figure_size=(12,12), save=True)




    if epoch % SAVE_EVERY_N_EPOCH == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch,
                                                             ckpt_save_path))
    
    D_lossl.append(-lde/(n_batches*N_CRITIC))
    G_lossl.append(lge/n_batches)
    Gradient_penalty.append(gpe/(n_batches*N_CRITIC))
    kll.append(kl)
    print('>%d, %d/%d, wdist=%.7f, g=%.7f gp=%.7f kl=%.7f' %
				(epoch+1, step+1, batch_per_epo, -lde/(n_batches*N_CRITIC), lge/n_batches, gpe/(n_batches*N_CRITIC),kl))
    print ('Time taken for epoch {} is {} minutes\n'.format(epoch,
                                                      (time.time()-start)/60))
if EPOCHs % SAVE_EVERY_N_EPOCH != 0: 
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(EPOCHs,
                                                        ckpt_save_path))



# # Save at final epoch
# # ckpt_save_path = ckpt_manager.save()
# #print ('Saving checkpoint for epoch {} at {}'.format(EPOCHs,ckpt_save_path))


# # In[21]:
if add_losses==True:
    results=np.load('results.npy')
    [D_lossl_o,G_lossl_o,kll_o,Gradient_penalty_o] = results
    print(results)

    D_lossl=np.append(D_lossl_o,D_lossl)
    G_lossl=np.append(G_lossl_o,G_lossl)
    kll=np.append(kll_o,kll)
    Gradient_penalty=np.append(Gradient_penalty_o,Gradient_penalty)
    

# # Use new sample to see the performance of the model.
plt.close("all")
plt.figure(figsize=(12,12))
# test_noise = tf.random.normal([8999216, NOISE_DIM])
# prediction = generator.predict(test_noise)

epochs_n=[i+1 for i in range (len(D_lossl))]
plt.plot(epochs_n,D_lossl,label="W distance")
plt.plot(epochs_n,G_lossl,label="Generator loss")
plt.plot(epochs_n,Gradient_penalty,label="Gradient penalty")
plt.ylabel('epochs')
plt.legend()
plt.savefig('loss_functions_and_gp.png')
plt.close("all")


plt.plot(epochs_n,D_lossl,label="W distance")
plt.plot(epochs_n,G_lossl,label="Generator loss")
plt.legend()
plt.ylabel('epochs')
plt.savefig('loss_functions.png')
plt.close("all")

plt.plot(epochs_n,D_lossl,label="W distance")
plt.plot(epochs_n,G_lossl,label="Generator loss")
plt.ylabel('epochs')
plt.legend()
plt.savefig('loss_functions.png')
plt.close("all")

plt.plot(epochs_n,kll,label="KL_divergence")
plt.yscale('log')
plt.ylabel('epochs')
plt.title('KL divergence')
plt.savefig('kl_divergence.png')


# def image_grid(images, fig):
#     # Create a figure to contain the plot.
#     for i in range(64):
#         # Start next subplot.
#         axs = fig.add_subplot(8, 8, i + 1)
#         axs.set_xticks([])
#         axs.set_yticks([])
#         axs.imshow(np.clip(images[i] * 0.5 + 0.5, 0, 1))
    
def save_plot(X,epoch):
    X=inverse_transform(X)

    hist=plt.hist2d(X[:,0],X[:,1],bins=60,cmap=cmap,range=[[-0.02,0.02],[-0.02,0.02]], vmin = 0.1)
    plt.colorbar()
    plt.suptitle(f'test noise results after epoch {epoch}')
    filename = 'test_noise.png' 
    plt.savefig(filename)
    plt.close()

results= [D_lossl,G_lossl,kll,Gradient_penalty]
results=np.array(results)
np.save('results',results)
print('losses, kl divergence and gp saved')


# Plot the real images for dataset
# plt.close("all")
# plt.figure(figsize=(12,12))
# save_plot(train_data)
# plt.savefig('example.png')

# plt.close("all")
# # Plot the fake images from the last epoch
# plt.figure(figsize=(12,12))
# save_plot(prediction,EPOCHs)


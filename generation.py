#


from __future__ import division

# SETTING CORE LIMITS FOR NUMPY AND PANDAS BEFORE IMPORTING THEM
import os
num_cores_str="30"
os.environ["OMP_NUM_THREADS"] = num_cores_str
os.environ["OPENBLAS_NUM_THREADS"] =num_cores_str
os.environ["MKL_NUM_THREADS"] = num_cores_str
os.environ["VECLIB_MAXIMUM_THREADS"] =num_cores_str 
os.environ["NUMEXPR_NUM_THREADS"] = num_cores_str

# SETTING GPU/CPU AND CORE LIMIT FOR TENSORFLOW
import tensorflow as tf

device='cpu'
if device =='cpu':
    tf.config.set_visible_devices([], 'GPU')

gpus = tf.config.get_visible_devices('GPU')

if gpus:
    print(f"GPUs available: {gpus}")
    print('Running on GPU')
else:
    print("No GPU found. Running on CPU.")
    num_cores=30
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)

# IMPORTING RELEVANT LIBRARIES
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
from scipy.stats import kstest as ks
import joblib
rng = np.random.default_rng()
cmap = plt.cm.viridis.copy()
cmap.set_under('black')  
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
features = 4
#ratio= signal/total
ratio=0.245345665
#names list to save the images and load quantile transformations
names=['signal','noise']




start=time.time()
#number of phonons taken as input
num_examples_to_generate = int(input("Number of phonons to generate : "))

#split the total number according to the ratio
split=[int(num_examples_to_generate*ratio), int(num_examples_to_generate*(1-ratio))]


# if the sum turns out to be less than num_examples to generate, that value is added to noise
if np.sum(split)<num_examples_to_generate:
    split[1]=split[1]+num_examples_to_generate-np.sum(split)




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


def load_real_samples(name):
    X=np.load(f'/home/ubuntu/Abhikamya/noise_removal/downconversion/{name}_1.npy')
 
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









#plotting function that takes a numpy array and 2 strings for title of model and type as input. 
def save_plot2(X,name, model):


    plt.figure(figsize=(30,25))

    plt.subplot(3,3,1,title='xy',xlabel='x in m', ylabel='y in m')
    hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,range=[[-0.02,0.02],[-0.02,0.02]],vmin=0.1)
    cbar1 = plt.colorbar()
    cbar1.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,2,title='xt full',xlabel='x in m', ylabel='t in ns')
    hist=plt.hist2d(X[:,0],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[mint,maxt]])
    plt.yscale('log')
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,3,title='xt zoom',xlabel='x in m', ylabel='t in ns')
    hist=plt.hist2d(X[:,0],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[0,5e+4]])
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)


    plt.subplot(3,3,4,title='yt full',xlabel='y in m', ylabel='t in ns')
    hist=plt.hist2d(X[:,1],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[mint,maxt]])
    plt.yscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,5,title='yt zoomed',xlabel='y in m', ylabel='t in ns')
    hist=plt.hist2d(X[:,1],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[0,5e+4]])
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)
    plt.suptitle('Signal for Initial Position X = 0')

    plt.subplot(3,3,6,title='et full',xlabel='e in ', ylabel='t in ns')
    hist=plt.hist2d(X[:,2],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[mine,maxe],[mint,maxt]])
    plt.yscale('log')
    plt.xscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,7,title='et zoomed',xlabel='e in ', ylabel='t in ns')
    hist=plt.hist2d(X[:,2],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[mine,maxe],[0,5e+4]])
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,8,title='xe',xlabel='x in m', ylabel='e in ')
    hist=plt.hist2d(X[:,0],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[mine,maxe]])
    plt.yscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,9,title='ye',xlabel='y in m', ylabel='e in ')
    hist=plt.hist2d(X[:,1],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[mine,maxe]])
    plt.yscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    


    plt.suptitle(f'{name} {model}')


    plt.savefig(f'{name}_{model}.png')
    plt.close()








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



discriminator = CGAN_discriminator()
D_optimizer = Adam(learning_rate=1e-05,     beta_1=0,
    beta_2=0.9)
G_optimizer = Adam(learning_rate=1e-05,     beta_1=0,
    beta_2=0.9)




#Empty list to combine noise and signal
final = []


for i in range(2):
    
    #generation is done seperately with the noise GAN and signal GAN, so this part of the code repeats for both noise and signal.  
    name=names[i]

    #load the minmax ranges that were saved previously. This allows to find these values without writing them directly into code or loading the original data. 
    [maxx,maxy,logmaxt,minx,miny,logmint,mint,maxt,mine,maxe,logmine,logmaxe]=list(np.load(f'/home/ubuntu/Abhikamya/Final/{name}_range_list.npy'))



    
    #load the Quantile transformations that were saved before. Again, this lets us generate new particles without loading the original dataset. 
    qtx=joblib.load(f'/home/ubuntu/Abhikamya/Final/Input_Processing/qtx_{name}')
    qty=joblib.load(f'/home/ubuntu/Abhikamya/Final/Input_Processing/qty_{name}')
    qte=joblib.load(f'/home/ubuntu/Abhikamya/Final/Input_Processing/qte_{name}')
    qtt=joblib.load(f'/home/ubuntu/Abhikamya/Final/Input_Processing/qtt_{name}')

    #restore the latest generator
    checkpoint_path = os.path.join(f"checkpoints_{name}", "tensorflow", MODEL_NAME)
    ckpt = tf.train.Checkpoint(generator=generator,discriminator=discriminator,G_optimizer=G_optimizer,D_optimizer=D_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,max_to_keep=None)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        latest_epoch = int(ckpt_manager.latest_checkpoint.split('-')[1])
        CURRENT_EPOCH = latest_epoch * SAVE_EVERY_N_EPOCH+1
        print ('Latest checkpoint of epoch {} restored!!'.format(CURRENT_EPOCH-1))
    
    
    #generate the number of particles needed in this set. This number can be found from the previous "split" list
    sample_noise = tf.random.normal([split[i], NOISE_DIM])

    #generate
    start_inside = time.time()
    fake=generator.predict(sample_noise)
    print ('Time taken for generating {} {} particles is {} minutes\n'.format(split[i], name,
                                                      (time.time()-start_inside)/60)

    #inverse transform the generated data
    start_inside2 = time.time()
    fake=inverse_transformation(fake)
    print ('Time taken for inverse transformation of {} {} particles is {} minutes\n'.format(split[i], name,
                                                      (time.time()-start_inside2)/60))
                                        

    #plot and save 2D histograms of the newly generated data
    save_plot2(fake,name,model='FastSim')

    #save the set
    np.save(f'FastSim_{name}',fake)
    print(f'{name} saved')

    #add the set to the final combined list
    final.extend(fake)

    #Delete everything created here to save memory. 
    del fake,sample_noise


print ('Total Time taken for generating and Saving {} particles is {} minutes\n'.format(num_examples_to_generate,
                                                      (time.time()-start)/60))

#plot the new particles
save_plot2(np.array(final),"Combined", model= 'FastSim')

#save the final array
np.save('FastSim_phonons_total',np.array(final))


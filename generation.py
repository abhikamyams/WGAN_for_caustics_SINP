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
import pandas as pd
import dask.dataframe as dd
import uproot
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
#set a limit for the total number of particles save in one file
limit=int(1e+7)


start=time.time()
#number of phonons taken as input
num_examples_to_generate = int(float((input("Number of phonons to generate : "))))
batches = [limit for i in range(int(num_examples_to_generate/limit))]
batches.append(num_examples_to_generate%limit)





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

generator.summary()
tf.keras.utils.plot_model(
    generator,
    to_file='generator.png',
    show_shapes=True,
    show_dtype=False,
    show_layer_names=False,
    rankdir='TB',
    expand_nested=False,
    dpi=200,
    show_layer_activations=False,
    show_trainable=False,

)

discriminator = CGAN_discriminator()


discriminator.summary()
tf.keras.utils.plot_model(
    discriminator,
    to_file='discriminator.png',
    show_shapes=True,
    show_dtype=False,
    show_layer_names=False,
    rankdir='TB',
    expand_nested=False,
    dpi=200,
    show_layer_activations=False,
    show_trainable=False,

)



D_optimizer = Adam(learning_rate=1e-05,     beta_1=0,
    beta_2=0.9)
G_optimizer = Adam(learning_rate=1e-05,     beta_1=0,
    beta_2=0.9)




#Empty list to combine noise and signal

filepath='/home/ubuntu/Abhikamya/Final/Training/WGAN26downconversion'


for j in range(len(batches)):
    final = []
    #split the total number according to the ratio
    split=[int(batches[j]*ratio), int(batches[j]*(1-ratio))]
        
    # if the sum turns out to be less than num_examples to generate, that value is added to noise
    if np.sum(split)<batches[j]:
        split[1]=split[1]+batches[j]-np.sum(split)
            
    for i in range(2):


        #generation is done seperately with the noise GAN and signal GAN, so this part of the code repeats for both noise and signal.  
        name=names[i]
        #load the minmax ranges that were saved previously. This allows to find these values without writing them directly into code or loading the original data. 
        [maxx,maxy,logmaxt,minx,miny,logmint,mint,maxt,mine,maxe,logmine,logmaxe]=list(np.load(f'/home/ubuntu/Abhikamya/Final/Input_Processing/{name}_range_list.npy',allow_pickle=True))
        
        #load the Quantile transformations that were saved before. Again, this lets us generate new particles without loading the original dataset. 
        qtx=joblib.load(f'/home/ubuntu/Abhikamya/Final/Input_Processing/qtx_{name}')
        qty=joblib.load(f'/home/ubuntu/Abhikamya/Final/Input_Processing/qty_{name}')
        qte=joblib.load(f'/home/ubuntu/Abhikamya/Final/Input_Processing/qte_{name}')
        qtt=joblib.load(f'/home/ubuntu/Abhikamya/Final/Input_Processing/qtt_{name}')
        
        #restore the latest generator
        checkpoint_path = os.path.join(filepath,f'{name}','checkpoints', "tensorflow", MODEL_NAME)
        print(checkpoint_path)
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
                                                      (time.time()-start_inside)/60))
                                                      
                                                      
        #inverse transform the generated data
        start_inside2 = time.time()
        fake=inverse_transformation(fake)
        print ('Time taken for inverse transformation of {} {} particles is {} minutes\n'.format(split[i], name,
                                                      (time.time()-start_inside2)/60))
            
            
        #plot and save 2D histograms of the newly generated data
        save_plot2(fake,name,model='FastSim')
            
        #add the set to the final combined list
        final.extend(fake)
            
        #Delete everything created here to save memory. 
        del fake
    final=np.array(final)
    save_plot2(final,"Before_cleanup", model= 'FastSim')



    #CLEANUP 
    start_cleanup= time.time()
    
    #load the valid 4dimensional bins that were saved during seperation. These contain the acceptable bins in which phonons can be expected in the dataset. 
    #Phonons that are not in any of these bins are purely due to 'leakage'
    bins = np.load('/home/ubuntu/Abhikamya/Final/Input_Processing/bins.npy',allow_pickle=True)
    
    #load the minmax ranges needed to apply these bins to the newly generated data
    [maxx,maxy,minx,miny,mint,maxt,mine,maxe]=list(np.load('/home/ubuntu/Abhikamya/Final/Input_Processing/full_range.npy',allow_pickle=True))
    
    #create sets of valid intervals from the bins
    bins = pd.DataFrame( data=bins, columns=['xbins', 'ybins', 'ebins','tbins'])
    valid_bins = pd.MultiIndex.from_frame(bins)
    #convert generated data into dask data frame.
    
    final=dd.from_array(final, columns=["Final_positionX", "Final_positionY", "Deposited_energy","Final_time"])
    
    #create bins from the loaded min/max ranges
    xbins=np.linspace(minx,maxx,101)
    ybins=np.linspace(miny,maxy,101)
    ebins=np.linspace(mine,maxe,101)
    tbins=np.linspace(mint,maxt,101)
    
    #bin the data in all 4 dimensions
    final['xbins']= final["Final_positionX"].map_partitions(pd.cut,xbins)
    final['ybins']= final["Final_positionY"].map_partitions(pd.cut,ybins)
    final['ebins']= final["Deposited_energy"].map_partitions(pd.cut,ebins)
    final['tbins']= final["Final_time"].map_partitions(pd.cut,tbins)
    
    
    #cleanup function
    def cleanup(df):
        #create intervals from bins in the generated data. 
        idx = pd.MultiIndex.from_frame(df[['xbins', 'ybins', 'ebins','tbins']])
        #isin works like a mask and will only return phonons within valid_bins
        return df[idx.isin(valid_bins)]
        
    #apply cleanup function to the dataset.
    final = np.array(final.map_partitions(cleanup, meta=final._meta).compute().drop(columns=['xbins','ybins', 'ebins','tbins']))
    
    print('Particles after cleanup',final.shape[0])
    
    np.save(f'FastSim_phonons_{j}',np.array(final))
    
    print ('Total Time taken for cleaning up {} particles is {} minutes\n'.format(final.shape[0],
                                                      (time.time()-start_cleanup)/60))

    #plot the particles after cleanup
    save_plot2(final,'After_cleanup', model = "FastSim")

    del final
    
print ('Total Time taken for generating and Saving {} particles is {} minutes\n'.format(num_examples_to_generate,
                                                      (time.time()-start)/60))






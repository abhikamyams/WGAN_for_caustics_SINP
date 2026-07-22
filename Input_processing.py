
# SETTING CORE LIMITS FOR NUMPY AND PANDAS BEFORE IMPORTING THEM
import os
num_cores_str="20"
os.environ["OMP_NUM_THREADS"] = num_cores_str
os.environ["OPENBLAS_NUM_THREADS"] =num_cores_str
os.environ["MKL_NUM_THREADS"] = num_cores_str
os.environ["VECLIB_MAXIMUM_THREADS"] =num_cores_str 
os.environ["NUMEXPR_NUM_THREADS"] = num_cores_str

# SETTING GPU/CPU AND CORE LIMIT FOR TENSORFLOW
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

# IMPORTING RELEVANT LIBRARIES
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import convert_to_tensor
tf.experimental.numpy.experimental_enable_numpy_behavior()
AUTOTUNE = tf.data.experimental.AUTOTUNE
from sklearn.preprocessing import QuantileTransformer 
rng = np.random.default_rng()
import joblib



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




def load_samples(name):

    #Load samples based on the name
    X=np.load(f'{name}_1.npy',allow_pickle=True)
    rng.shuffle(X, axis=0)
    X=X.astype(np.float32)

    #define min/max ranges for regularization
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



#for loop to run for both noise and signal
for i in range(2):

    #seperate Quantile transformations loaded
    qtx= QuantileTransformer(output_distribution='normal', n_quantiles=100000,subsample=20000000)
    qty= QuantileTransformer(output_distribution='normal', n_quantiles=100000,subsample=20000000)
    qte= QuantileTransformer(output_distribution='normal', n_quantiles=100000,subsample=20000000)
    qtt= QuantileTransformer(output_distribution='normal', n_quantiles=100000,subsample=20000000)

    
    #names for saving
    if i==0:
        name='signal'
    else:
        name='noise'
    
    #load data as X
    X,maxx,maxy,logmaxt,minx,miny,logmint,mint,maxt,mine,maxe,logmine,logmaxe=load_samples(name)

    #transform X to X_tr
    X_tr=transformation(X)

    #save the ranges and the transformed set
    range_list = np.array([maxx,maxy,logmaxt,minx,miny,logmint,mint,maxt,mine,maxe,logmine,logmaxe])
    np.save(f'{name}_range_list',range_list)
    np.save(f'{name}_transformed',X_tr)

    #save the Quantile transformations
    joblib.dump(qtx,f'qtx_{name}')
    joblib.dump(qty,f'qty_{name}')
    joblib.dump(qte,f'qte_{name}')
    joblib.dump(qtt,f'qtt_{name}')

    del X, X_tr



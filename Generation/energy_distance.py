
import os
num_cores_str="30"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = num_cores_str
os.environ["OPENBLAS_NUM_THREADS"] =num_cores_str
os.environ["MKL_NUM_THREADS"] = num_cores_str
os.environ["VECLIB_MAXIMUM_THREADS"] =num_cores_str 
os.environ["NUMEXPR_NUM_THREADS"] = num_cores_str

print(os.environ["OMP_NUM_THREADS"])
print(os.environ["OPENBLAS_NUM_THREADS"])
print(os.environ["MKL_NUM_THREADS"])
import numpy as np
np.show_config()
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp as ks
import matplotlib.pyplot as plt
import dask.dataframe as dd
import time
rng = np.random.default_rng()
features=2
batch_size=10000

#load FastSim data
FastSim=np.load("FastSim_phonons_total.npy")

#load FullSim data
FullSim=dd.read_parquet('/home/ubuntu/Abhikamya/Original_root_files/Output1.parquet', 
columns=["Final_positionX", "Final_positionY","Deposited_energy", "Final_time"])


#compute only the same amount of particles from fullsim
FullSim=np.array(FullSim.head(FastSim.shape[0],npartitions=2))
names=['X','Y','E','T']

#Comparision histograms
def save_plot(model1,model2, n1,n2,save):
    for i in range(4):
        name=names[i]
        plt.subplot(2,1,1)
        Xlf=plt.hist(model1[:,i],label=f"{n1} {name}",alpha=0.5,bins=100)
        Xl=plt.hist(model2[:,i],label=f"{n2} {name}",alpha=0.5,bins=100)
        plt.legend()
        plt.subplot(2,1,2)
        n=[]
        for j in range(len(Xlf[0])):
            if Xl[0][j]==0:
                n.append(0)
            else:
                n.append(Xlf[0][j]/Xl[0][j])
        plt.stairs(n,edges=Xlf[1])
        plt.suptitle(f'{name} {n1}Vs{n2} {save}')
        plt.savefig(f'{name}{save}.png')
        plt.close('all')

save_plot(FullSim,FastSim,'FullSim','FastSim','before_regularization')



#get minmax ranges
maxx=np.max(FullSim[:,0])
minx=np.min(FullSim[:,0])

maxy=np.max(FullSim[:,1]) 
miny=np.min(FullSim[:,1])

maxe=np.max(FullSim[:,2]) 
mine=np.min(FullSim[:,2]) 


mint=np.min(FullSim[:,3])
maxt=np.max(FullSim[:,3])
    
logmint=np.min(np.log(FullSim[:,3]))
logmaxt=np.max(np.log(FullSim[:,3]))

logmine=np.min(np.log(FullSim[:,2]))
logmaxe=np.max(np.log(FullSim[:,2]))


print(np.shape(FullSim))
print(np.shape(FastSim))

#regularization
def regularization(Y):
    X=Y.copy()

    #linear normalization
    X[:,0]=(X[:,0] - minx)/(maxx-minx)
    X[:,1]=(X[:,1] - miny)/(maxy-miny)
    
    #log then linear normalization
    X[:,2]=(np.log(X[:,0])-logmine)/(logmaxe-logmine)
    X[:,3]=(np.log(X[:,0])-logmint)/(logmaxt-logmint)
    return X

#energy distance
def get_energy_distance(x,y):
    n=x.shape[0]
    #cdist computes euclidean distance which will be a matrix of shape (n,n)
    xx=cdist(x,x)
    yy=cdist(y,y)
    xy=cdist(x,y)
    return (1/n**2)*np.sum((2*xy-xx-yy))

def get_bed(X,Y):

    #get the number of batches
    b=int(FastSim.shape[0]/batch_size)

    real=X.copy()
    fake=Y.copy()
    print(np.shape(real))

    #regularization
    fake,real=regularization(fake),regularization(real)

    #split the arrays according to batches
    fake=np.array_split(fake,b)
    real=np.array_split(real,b)

    #get indexes for the batches
    inx=np.arange(b)

    #if there are not enough elements in the last batch, remove the last index from the index to avoid ever calling it
    if fake[b-1].shape[0] != batch_size:
        inx=np.arange(b-1)
        b=b-1

    #create 2 index lists with a random orders. if one is not random and fake=real, then the null case would be zero. 
    inxo=np.arange(b)
    inx=np.arange(b)
    rng.shuffle(inx)
    rng.shuffle(inxo)

    #shuffle the batches 
    real2=np.array([real[i] for i in inx])
    real=np.array([real[i] for i in inxo])
    fake=np.array([fake[i] for i in range(b)])

    h0=[]
    h1=[]
    k=1
    for i in range(b):
        #get energy distance of null case
        h0.append(get_energy_distance(real[i],real2[i]))
        #get energy distance of test case
        h1.append(get_energy_distance(real2[i],fake[i]))
        print(k)
        k=k+1
    print(np.max(h0))
    print(np.max(h1))
    #plot energy distance histograms
    plt.hist(h0,label='null set',alpha =0.5,bins=100)
    plt.hist(h1,label='test set',alpha = 0.5,bins=100)
    plt.title('Batched Energy Distance')
    plt.legend()
    plt.savefig('BED.png')
    
    #return the results of the K-S test 
    return ks(h0,h1)

#if you need to get energy distance for specific axes, call them using numpy slicing. eg.get_bed(FullSim[:,1:3],FastSim[:,1:3]) --> which would test for only y axis and energy
print("Result of Batched energy Distance Calculation", get_bed(FullSim,FastSim))


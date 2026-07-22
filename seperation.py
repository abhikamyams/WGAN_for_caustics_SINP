#To seperate signal from noise: 


# This code only works when you want to remove noise from particles less than 2 x 10^8, if you go upto 8 x 10e+8 it will hit a memory limit. This is because the noise and signal files are kept as
# numpy files in the ram before saving. You can fix this by having the code write directly into zarr files instead of storing the numpy file in ram before saving. 
# I tried this and there was an error where the expected number of particles were not found in the noise or signal file. Since we only need to work with 10e+7 particles anyways
# (partly for computational speed, partly for limiting memory usage), I didnt look into debugging that.
# Im sure with dask there will be more efficient ways to execute this. For the current data set,this code is sufficient.


import os
num_cores_str="25"
os.environ["OMP_NUM_THREADS"] = num_cores_str
os.environ["OPENBLAS_NUM_THREADS"] =num_cores_str
os.environ["MKL_NUM_THREADS"] = num_cores_str
os.environ["VECLIB_MAXIMUM_THREADS"] =num_cores_str 
os.environ["NUMEXPR_NUM_THREADS"] = num_cores_str


import numpy as np
import pandas as pd
import zarr
import dask
dask.config.set({"scheduler": "threads", "num-workers": 8})
import dask.dataframe as dd
import dask.array as da
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import time
rng = np.random.default_rng()
cmap = plt.cm.viridis.copy()
cmap.set_under('black')  




#define the number of bins made in each axis
nbins=100

#define the number of particles to remove from each bin
remove= 20000

train_data=dd.read_parquet('/home/ubuntu/Abhikamya/Original_root_files/Output1.parquet', 
columns=["Final_positionX", "Final_positionY","Deposited_energy", "Final_time"])

print('Number of particles before noise removal : ',len(train_data))

print('no of partitions in dataset', train_data.npartitions)

#Save the Columns as dask arrays for plotting
X=train_data['Final_positionX'].to_dask_array()
Y=train_data['Final_positionY'].to_dask_array()
E=train_data['Deposited_energy'].to_dask_array()
T=train_data['Final_time'].to_dask_array()

#define min/max ranges of the dataset.
maxx=np.max(train_data['Final_positionX']).compute()
minx=np.min(train_data['Final_positionX']).compute()

maxy=np.max(train_data['Final_positionY']).compute()
miny=np.min(train_data['Final_positionY']).compute()

maxt=np.max(train_data['Final_time']).compute()
mint=np.min(train_data['Final_time']).compute()

maxe=np.max(train_data['Deposited_energy']).compute()
mine=np.min(train_data['Deposited_energy']).compute()

np.save('full_range',np.array([maxx,maxy,minx,miny,mint,maxt,mine,maxe]))




#create edges for bins
xbins=np.linspace(minx,maxx,101)
ybins=np.linspace(miny,maxy,101)
ebins=np.linspace(mine,maxe,101)
tbins=np.linspace(mint,maxt,101)



#Save plots of the original FullSim data
def save_plot():

    plt.figure(figsize=(30,25))
    plt.subplot(3,3,1,title='XY',xlabel='X in m', ylabel='Y in m')
    hist,_=da.histogramdd((X,Y),bins=100,range=((minx, maxx),(miny, maxy)),)
    plt.imshow(hist.T, origin='lower',extent=[minx, maxx, miny, maxy],cmap=cmap,vmin=0.1)
    cbar1 = plt.colorbar()
    cbar1.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,2,title='X-T full',xlabel='X in m', ylabel='Time in ns')
    hist,_=da.histogramdd((X,T),bins=100,range=((minx, maxx),(mint,maxt)))
    plt.imshow(hist.T, origin='lower',extent=[minx, maxx, mint, maxt],cmap=cmap,vmin=0.1,aspect='auto')
    plt.yscale('log')
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,3,title='X-T zoom',xlabel='X in m', ylabel='Time in ns')
    hist,_=da.histogramdd((X,T),bins=100,range=((minx, maxx),(0,5e+4)))
    plt.imshow(hist.T, origin='lower',extent=[minx, maxx, 0,5e+4],cmap=cmap,vmin=0.1,aspect='auto')
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,4,title='Y-T full',xlabel='Y in m', ylabel='Time in ns')
    hist,_=da.histogramdd((Y,T),bins=100,range=((miny, maxy),(mint,maxt)))
    plt.imshow(hist.T, origin='lower',extent=[miny, maxy, mint, maxt],cmap=cmap,vmin=0.1,aspect='auto')
    plt.yscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,5,title='Y-T zoomed',xlabel='Y in m', ylabel='Time in ns')
    hist,_=da.histogramdd((Y,T),bins=100,range=((miny, maxy),(0,5e+4)))
    plt.imshow(hist.T, origin='lower',extent=[miny, maxy, 0,5e+4],cmap=cmap,vmin=0.1,aspect='auto')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)
    plt.suptitle('Full set for Initial Position X = 0')

    plt.subplot(3,3,6,title='E-T full',xlabel='Energy in eV', ylabel='Time in ns')
    hist,_=da.histogramdd((E,T),bins=100,range=((mine,maxe),(mint,maxt)))
    plt.imshow(hist.T, origin='lower',extent=[mine, maxe, mint, maxt],cmap=cmap,vmin=0.1,aspect='auto')
    plt.yscale('log')
    plt.xscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)


    plt.subplot(3,3,7,title='E-T zoomed',xlabel='Energy in eV', ylabel='Time in ns')
    hist,_=da.histogramdd((E,T),bins=100,range=((mine,maxe),(0,5e+4)))
    plt.imshow(hist.T, origin='lower',extent=[mine, maxe, 0,5e+4],cmap=cmap,vmin=0.1,aspect='auto')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,8,title='X-E',xlabel='X in m', ylabel='Energy in eV ')
    hist,_=da.histogramdd((X,E),bins=100,range=((minx, maxx),(mine,maxe)))
    plt.imshow(hist.T, origin='lower',extent=[minx, maxx, mine,maxe],cmap=cmap,vmin=0.1,aspect='auto')
    plt.yscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,9,title='Y-E',xlabel='Y in m', ylabel='Energy in eV ')
    hist,_=da.histogramdd((Y,E),bins=100,range=((miny, maxy),(mine,maxe)))
    plt.imshow(hist.T, origin='lower',extent=[miny, maxy, mine,maxe],cmap=cmap,vmin=0.1,aspect='auto')
    plt.yscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.suptitle('Before Seperation')


    filename = 'FullSim_downconversion.png'
    plt.savefig(filename)
    plt.close()

save_plot()

#Bin the data according to xbins, ybins, ebins and tbins from before
train_data['xbins']= train_data["Final_positionX"].map_partitions(pd.cut,xbins)
train_data['ybins']= train_data["Final_positionY"].map_partitions(pd.cut,ybins)
train_data['ebins']= train_data["Deposited_energy"].map_partitions(pd.cut,ebins)
train_data['tbins']= train_data["Final_time"].map_partitions(pd.cut,tbins)

#save all the valid 4dimensional bin combinations in which particles can be found in the training data
bins = (
    train_data[["xbins", "ybins","ebins","tbins"]]
    .drop_duplicates()
    .compute()
)
np.save('bins',bins)


#define empty bins for signal/clean and noise
clean = []
noise = []


#function for seperation
def d(x):
    #get the index of one group (which would mean one bin in XY).
    index=x.index
    

    #check if there are more particles than we want to remove
    if len(index)>remove:
        #move the extra particles to clean and the initial constant amount of particles to noise
        clean.extend(x.iloc[remove:].to_numpy())
        noise.extend(x.iloc[:remove].to_numpy())
    else:
        #move the whole bin to noise
        noise.extend(x.to_numpy())

    #return an empty data frame. This is needed because dask cannot do group by. apply on functions that do not have anything to return. 
    return pd.DataFrame(columns=x.columns)





start=time.time()
#group the dataset by xbins and ybins, then apply d to each group
del_grouped=np.array(train_data.groupby(['xbins', 'ybins']).apply(d,meta=train_data._meta).compute())
del train_data,del_grouped


print('Number of signal particles : ',np.shape(clean))
print('Number of noise particles  : ',np.shape(noise))

clean=np.array(clean)
noise=np.array(noise)

#shuffling the array is necessary because without it the array is ordered by x and y bins. 
rng.shuffle(clean, axis=0)
rng.shuffle(noise, axis=0)

#save the noise and signal sets as numpy files. 
np.save('signal_1',clean)
np.save('noise_1',noise)


print ('Time taken for noise removal is {} minutes\n'.format((time.time() - start)/60))



def save_plot2(X,Name):
    
    plt.figure(figsize=(30,25))

    plt.subplot(3,3,1,title='X-Y',xlabel='X in m', ylabel='Y in m')
    hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,range=[[-0.02,0.02],[-0.02,0.02]],vmin=0.1)
    cbar1 = plt.colorbar()
    cbar1.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,2,title='X-T full',xlabel='X in m', ylabel='Time in ns')
    hist=plt.hist2d(X[:,0],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[mint,maxt]])
    plt.yscale('log')
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,3,title='X-T zoom',xlabel='X in m', ylabel='Time in ns')
    hist=plt.hist2d(X[:,0],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[0,5e+4]])
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)


    plt.subplot(3,3,4,title='Y-T full',xlabel='Y in m', ylabel='Time in ns')
    hist=plt.hist2d(X[:,1],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[mint,maxt]])
    plt.yscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,5,title='Y-T zoomed',xlabel='Y in m', ylabel='Time in ns')
    hist=plt.hist2d(X[:,1],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[0,5e+4]])
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)
    plt.suptitle('Signal for Initial Position X = 0')

    plt.subplot(3,3,6,title='E-T full',xlabel='Energy in eV', ylabel='Time in ns')
    hist=plt.hist2d(X[:,2],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[mine,maxe],[mint,maxt]])
    plt.yscale('log')
    plt.xscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,7,title='E-T zoomed',xlabel='Energy in eV ', ylabel='Time in ns')
    hist=plt.hist2d(X[:,2],X[:,3],bins=100,cmap=cmap,vmin=0.1,range=[[mine,maxe],[0,5e+4]])
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,8,title='X-E',xlabel='X in m', ylabel='Energy in eV ')
    hist=plt.hist2d(X[:,0],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[mine,maxe]])
    plt.yscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(3,3,9,title='Y-E',xlabel='Y in m', ylabel='Energy in eV ')
    hist=plt.hist2d(X[:,1],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[mine,maxe]])
    plt.yscale('log')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)




    plt.suptitle(f'{Name} set FullSim')
    plt.savefig(f'{Name}_FullSim.png')
    plt.close()

#plot signal and noise
save_plot2(clean,'Signal')

save_plot2(noise,'Noise')

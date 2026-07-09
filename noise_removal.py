


# This code only works when you want to remove noise from particles in order 10e+7, if you go upto 10e+8 it will hit a memory limit. You can fix this by having it right directly 
# into zarr files instead of storing the numpy file in ram before saving. I tried this and there was an error where the expected number of particles were not found in the noise or signal file. 
# since we only need to work with 10e+7 particles anyways (partly for computational speed, partly for limiting memory usage), I didnt look into debugging that. Im sure with dask there will be more 
# efficient ways to execute this.

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

# import hepconvert 

# hepconvert.root_to_parquet('/home/ubuntu/Abhikamya/Original_root_files/10M_N_Caustics_1p5_x.root','/home/ubuntu/Abhikamya/Original_root_files/10M_N_Caustics_1p5_x.parquet',tree="Per_event_data", keep_branches=["Final_positionX","Final_positionY","Final_time"])


train_data=dd.read_parquet('/home/ubuntu/Abhikamya/Original_root_files/10M_N_Caustics_1p5_x.parquet', 
columns=["Final_positionX", "Final_positionY", "Final_time"])

print('Number of particles before noise removal : ',len(train_data))

print('no of partitions in dataset', train_data.npartitions)

plt.figure(figsize=(10,10))
X= train_data['Final_positionX'].to_dask_array()
Y= train_data['Final_positionY'].to_dask_array()
T= train_data['Final_time'].to_dask_array()

xmax=np.max(train_data['Final_positionX']).compute()
xmin=np.min(train_data['Final_positionX']).compute()

ymax=np.max(train_data['Final_positionY']).compute()
ymin=np.min(train_data['Final_positionY']).compute()

tmax=np.max(train_data['Final_time']).compute()
tmin=np.min(train_data['Final_time']).compute()


xbins=np.linspace(xmin,xmax,101)
ybins=np.linspace(ymin,ymax,101)


# histx,_ = da.histogram(X,range=(-0.02,0.02),bins=100)

# plt.figure(figsize=(6,6))
# plt.stairs(histx,xbins)
# plt.title('x for X =1.5 Initial Position')
# plt.savefig('fullset_x-1p5.png')
# plt.close()

# print('x plot done')

# histy,_ = da.histogram(Y,range=(-0.02,0.02),bins=100)

# plt.figure(figsize=(6,6))
# plt.stairs(histy,ybins)
# plt.title('y for X =1.5 Initial Position')
# plt.savefig('y-1p5.png')
# plt.close()

# print('y plot done')
xmin=-0.02
xmax=0.02
ymin=-0.02
ymax=0.02

def save_plot():
    plt.figure(figsize=(25,10))
    

    plt.subplot(2,3,1,title='xy',xlabel='x in m', ylabel='y in m')
    hist,_=da.histogramdd((X,Y),bins=100,range=((-0.02,0.02),(-0.02,0.02)),)
    plt.imshow(hist.T, origin='lower',extent=[xmin, xmax, ymin, ymax])
    cbar1 = plt.colorbar()
    cbar1.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(2,3,2,title='xt full',xlabel='x in m', ylabel='t in ns')
    hist,_=da.histogramdd((X,T),bins=100,range=((-0.02,0.02),(tmin,tmax)))
    plt.imshow(hist.T, origin='lower',extent=[xmin, xmax, tmin, tmax],cmap=cmap,vmin=0.1,aspect='auto')
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(2,3,3,title='xt zoom',xlabel='x in m', ylabel='t in ns')
    hist,_=da.histogramdd((X,T),bins=100,range=((-0.02,0.02),(0,5e+4)))
    plt.imshow(hist.T, origin='lower',extent=[xmin, xmax, 0,5e+4],cmap=cmap,vmin=0.1,aspect='auto')
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)


    plt.subplot(2,3,4,title='yt full',xlabel='y in m', ylabel='t in ns')
    hist,_=da.histogramdd((Y,T),bins=100,range=((-0.02,0.02),(tmin,tmax)))
    plt.imshow(hist.T, origin='lower',extent=[xmin, xmax, tmin, tmax],cmap=cmap,vmin=0.1,aspect='auto')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(2,3,5,title='yt zoomed',xlabel='y in m', ylabel='t in ns')
    hist,_=da.histogramdd((Y,T),bins=100,range=((-0.02,0.02),(0,5e+4)))
    plt.imshow(hist.T, origin='lower',extent=[xmin, xmax, 0,5e+4],cmap=cmap,vmin=0.1,aspect='auto')
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)
    plt.suptitle('Full set for Initial Position X = 1.5')

    # plt.subplots_adjust(left=0.15, bottom=0.15, right=0.82, top=0.85)
    filename = 'fullset_1p5.png'
    plt.savefig(filename)
    plt.close()

save_plot()
start = time.perf_counter()



train_data['xbins']= train_data["Final_positionX"].map_partitions(pd.cut,xbins)
train_data['ybins']= train_data["Final_positionY"].map_partitions(pd.cut,xbins)
bins = train_data['xbins'].cat.categories
# print(len(bins))

clean = []
noise = []
remove= 10000
i=0
j=0
# k=0
# bins=[]
# bins_after=[]
def d(x):
    global j
    global i
    # global k
    # global bins
    # global bins_after
    index=x.index
    # bins.append(len(x))

    if len(index)>remove:
        # np.concatenate((clean,x.iloc[remove:].to_numpy()))
        # np.concatenate((noise,x.iloc[:remove].to_numpy()))
        
        clean.extend(x.iloc[remove:].to_numpy())
        noise.extend(x.iloc[:remove].to_numpy())
        
        return pd.DataFrame(columns=x.columns)
    else:
        
        # np.concatenate((noise,x.to_numpy()))
        noise.extend(x.to_numpy())
        return pd.DataFrame(columns=x.columns)
        # clean.append(pd.DataFrame(columns=x.columns))
        # return pd.DataFrame(columns=x.columns)


meta= {
    "Final_positionX": "int64",
    "Final_positionY": "float64",
    "Final_time": "float64"
    
}


print('before noise removal')

del_grouped=np.array(train_data.groupby(['xbins', 'ybins']).apply(d,meta=meta).compute())

# clean=np.load('/home/ubuntu/Abhikamya/noise_removal/noise_removal_2GANS/without_noise_100000.npy')
# noise=np.load('/home/ubuntu/Abhikamya/noise_removal/noise_removal_2GANS/noise_100000.npy')
print(np.shape(clean))
print(np.shape(noise))
rng.shuffle(clean, axis=0)
rng.shuffle(noise, axis=0)
np.save('signal_1p5',clean)
np.save('noise_1p5',noise)
print(i,j,i+j)
# print('total number of bins before',np.sum(bins))
# print('total number of bins after',np.sum(bins_after))
# print(bins)
# print(bins_after)
# print(np.shape(del_grouped))
#np.save("without_noise_100000",del_grouped)
print ('Time taken for noise removal is {} minutes\n'.format(
                                                      (time.perf_counter()-start)/60))
print('noise removal done')

print('what happened to del grouped',len(del_grouped))

print('Number of particles after noise removal : ',np.shape(clean))

print('Number of noise particles that were removed : ',np.shape(noise))


# WITHOUT NOISE SECTION

print(np.shape(clean))

# del_grouped=np.load("without_noise_100000.npy")
# print(np.shape(del_grouped))
# del del_grouped
rng.shuffle(clean, axis=0)

def save_plot(X):
    plt.figure(figsize=(30,10))
    X=np.array(X)
    plt.subplot(2,3,1,title='xy',xlabel='x in m', ylabel='y in m')
    hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,range=[[-0.02,0.02],[-0.02,0.02]],vmin=0.1)
    cbar1 = plt.colorbar()
    cbar1.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(2,3,2,title='xt full',xlabel='x in m', ylabel='t in s')
    hist=plt.hist2d(X[:,0],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[tmin,tmax]])
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(2,3,3,title='xt zoom',xlabel='x in m', ylabel='t in s')
    hist=plt.hist2d(X[:,0],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[0,5e+4]])
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)


    plt.subplot(2,3,4,title='yt full',xlabel='y in m', ylabel='t in s')
    hist=plt.hist2d(X[:,1],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[tmin,tmax]])
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(2,3,5,title='yt zoomed',xlabel='y in m', ylabel='t in s')
    hist=plt.hist2d(X[:,1],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[0,5e+4]])
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)
    plt.suptitle('Signal for Initial Position X = 1.5')

    # plt.subplots_adjust(left=0.15, bottom=0.15, right=0.82, top=0.85)
    filename = 'signal_1p5.png'
    plt.savefig(filename)
    plt.close()

save_plot(clean)

def save_plot(X):
    plt.figure(figsize=(30,10))
    X=np.array(X)

    plt.subplot(2,3,1,title='xy',xlabel='x in m', ylabel='y in m')
    hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,range=[[-0.02,0.02],[-0.02,0.02]],vmin=0.1)
    cbar1 = plt.colorbar()
    cbar1.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(2,3,2,title='xt full',xlabel='x in m', ylabel='t in s')
    hist=plt.hist2d(X[:,0],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[tmin,tmax]])
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(2,3,3,title='xt zoom',xlabel='x in m', ylabel='t in s')
    hist=plt.hist2d(X[:,0],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[0,5e+4]])
    cbar2 = plt.colorbar()
    cbar2.set_label('Counts in each pixel', labelpad=15, fontsize=11)


    plt.subplot(2,3,4,title='yt full',xlabel='y in m', ylabel='t in s')
    hist=plt.hist2d(X[:,1],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[tmin,tmax]])
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)

    plt.subplot(2,3,5,title='yt zoomed',xlabel='y in m', ylabel='t in s')
    hist=plt.hist2d(X[:,1],X[:,2],bins=100,cmap=cmap,vmin=0.1,range=[[-0.02,0.02],[0,5e+4]])
    cbar3 = plt.colorbar()
    cbar3.set_label('Counts in each pixel', labelpad=15, fontsize=11)
    plt.suptitle('Noise for Initial Position X = 1.5')

    # plt.subplots_adjust(left=0.15, bottom=0.15, right=0.82, top=0.85)
    filename = 'noise_1p5.png'
    plt.savefig(filename)
    plt.close()

save_plot(noise)


# # def save_plot(X):
# #     plt.figure(figsize=(6,6))
# #     print('Number of particles in one batch : ',len(X))
# #     hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,vmin=0.01)
# #     plt.xlabel('x in cms', fontsize=11)
# #     plt.ylabel('y in cms', fontsize=11)
# #     plt.title('128*128*4_noise', pad=15)
# #     cbar = plt.colorbar()
# #     cbar.set_label('Counts in each pixel', labelpad=15, fontsize=11)
# #     plt.subplots_adjust(left=0.15, bottom=0.15, right=0.82, top=0.85)
# #     plt.savefig('128*128*4_noise.png')
# #     plt.close()

# # save_plot(del_grouped[:128*128*4,:])


# # #NOISE SECTION


# # def save_plot(X):
# #     plt.figure(figsize=(6,6))
# #     hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,vmin=0.01)
# #     plt.title('xy_noise')
# #     plt.colorbar()
# #     plt.savefig('xy_noise.png')
# #     plt.close()
# # save_plot(noise)

# # def save_plot(X):
# #     plt.figure(figsize=(6,6))
# #     print(np.shape(X[:,0]))
# #     hist,xedges,_=plt.hist(X[:,0],bins=100)
# #     plt.title('x_noise')
# #     plt.savefig('x_noise.png')
# #     plt.close()
# #     return 
# # save_plot(noise)

# # def save_plot(X):
# #     plt.figure(figsize=(6,6))

# #     hist,yedges,_=plt.hist(X[:,1],bins=100)
# #     plt.title('y_noise')
# #     plt.savefig('y_noise.png')
# #     plt.close()
# #     return

# # save_plot(noise)





# # def save_plot(X):
# #     plt.figure(figsize=(6,6))
# #     print('Number of particles in one batch : ',len(X))
# #     hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,vmin=0.01)
# #     plt.title('128*128*4 noise')
# #     plt.colorbar()
# #     plt.savefig('128*128*4_noise.png')
# #     plt.close()

# # save_plot(noise[:128*128*4,:])






























# def save_plot(X):
#     plt.figure(figsize=(6,6))
#     print(np.shape(X[:,0]))
#     hist,xedges,_=plt.hist(X[:,0],bins=100)
#     plt.title('x')
#     plt.savefig('x.png')
#     plt.close()
#     return xedges
# xedges=save_plot(train_data)
# def save_plot(X):
#     plt.figure(figsize=(6,6))
#     print(np.shape(X[:,1]))
#     hist,yedges,_=plt.hist(X[:,1],bins=100)
#     plt.title('y')
#     plt.savefig('y.png')
#     plt.close()
#     return yedges
# yedges=save_plot(train_data)
# def save_plot(X):
#     plt.figure(figsize=(6,6))
#     print(np.shape(X[:,1]))
#     hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,vmin=0.01)
#     plt.title('xy')
#     plt.colorbar()
#     plt.savefig('xy.png')
#     plt.close()



# save_plot(train_data)
# remove=800


# xbins=np.linspace(np.min(train_data[:,0]),np.max(train_data[:,0]),101)
# ybins=np.linspace(np.min(train_data[:,1]),np.max(train_data[:,1]),101)
# train_data=pd.DataFrame(data=train_data, columns=['x','y','e','t'])
# train_data['xbins']=pd.cut(train_data['x'],xbins)
# train_data['ybins']=pd.cut(train_data['y'],ybins)
# bins = train_data['xbins'].cat.categories

# grouped = train_data.groupby(['xbins', 'ybins'])
# def d(x):
#     index=x.index

#     if len(index)>remove:
#         x=x.drop(index[:remove])
#     else:
#         x=x.drop(index)
#     return x

# del_grouped=grouped.apply(d)

# print('Number of particles after noise removal : ',len(del_grouped))

# new_data=np.array(del_grouped)
# rng.shuffle(new_data, axis=0)
# def save_plot(X):
#     plt.figure(figsize=(6,6))
#     hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,vmin=0.01)
#     plt.title('xy_noise_removed')
#     plt.colorbar()
#     plt.savefig('xy_noise_removed.png')
#     plt.close()
# save_plot(new_data)

# def save_plot(X):
#     plt.figure(figsize=(6,6))
#     print(np.shape(X[:,0]))
#     hist,xedges,_=plt.hist(X[:,0],bins=100)
#     plt.title('x_noise_removed')
#     plt.savefig('x_noise_removed.png')
#     plt.close()
#     return 
# save_plot(new_data)

# def save_plot(X):
#     plt.figure(figsize=(6,6))

#     hist,yedges,_=plt.hist(X[:,1],bins=100)
#     plt.title('y_noise_removed')
#     plt.savefig('y_noise_removed.png')
#     plt.close()
#     return

# save_plot(new_data)


# def save_plot(X):
#     plt.figure(figsize=(6,6))
#     print('Number of particles in one batch : ',len(X))
#     hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,vmin=0.01)
#     plt.title('128*128*8')
#     plt.colorbar()
#     plt.savefig('128*128*8.png')
#     plt.close()

# save_plot(new_data[:128*128*8,:])



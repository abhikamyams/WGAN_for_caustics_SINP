
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

train_data=dd.read_parquet('/home/ubuntu/Abhikamya/Original_root_files/Without_downconversion.parquet', 
columns=["Final_positionX", "Final_positionY", "Deposited_energy", "Final_time"])

print('Number of particles before noise removal : ',len(train_data))

print('no of partitions in dataset', train_data.npartitions)

# X= train_data['Final_positionX'].to_dask_array()
# Y= train_data['Final_positionY'].to_dask_array()


xmax=np.max(train_data['Final_positionX']).compute()
xmin=np.min(train_data['Final_positionX']).compute()

ymax=np.max(train_data['Final_positionY']).compute()
ymin=np.min(train_data['Final_positionY']).compute()

xbins=np.linspace(xmin,xmax,101)
ybins=np.linspace(ymin,ymax,101)


# histx,_ = da.histogram(X,range=(-0.02,0.02),bins=100)

# plt.figure(figsize=(6,6))
# plt.stairs(histx,xbins)
# plt.title('x')
# plt.savefig('x.png')
# plt.close()

# print('x plot done')

# histy,_ = da.histogram(Y,range=(-0.02,0.02),bins=100)

# plt.figure(figsize=(6,6))
# plt.stairs(histy,ybins)
# plt.title('y')
# plt.savefig('y.png')
# plt.close()

# print('y plot done')

# histxy,_ ,_= da.histogram2d(X,Y,range=((-0.02,0.02),(-0.02,0.02)),bins=100)


# plt.figure(figsize=(6,6))
# plt.imshow(histxy,cmap=cmap,vmin=0.1)
# plt.title('xy full set before noise removal')
# plt.colorbar()
# plt.savefig('xy.png')
# plt.close()

# print('xy plot done')


start = time.perf_counter()



train_data['xbins']= train_data["Final_positionX"].map_partitions(pd.cut,xbins)
train_data['ybins']= train_data["Final_positionY"].map_partitions(pd.cut,xbins)

# clean = zarr.open('without_noise_100000', mode='w', shape=(0, 4), chunks=(100000, 4), dtype='float64')
# noise = zarr.open('noise_100000.zarr', mode='w', shape=(0, 4), chunks=(100000, 4), dtype='float64')

remove= 100000
i=0
j=0
k=0
bins=[]
bins_after=[]
def d(x):
    global j
    global i
    global k
    global bins
    index=x.index
    bins.append(len(x))
    if len(index)>remove:
        k=k+1
        x=x.drop(index[:remove])
        i=i+len(x.index)
    else:
        j=j+len(index)
        x=x.drop(index)
    bins_after.append(len(x))
    return x
meta= {
    "Final_positionX": "int64",
    "Final_positionY": "float64",
    "Deposited_energy": "float64",
    "Final_time": "float64"
    
}

print('before noise removal')

del_grouped=np.array(train_data.groupby(['xbins', 'ybins']).apply(d,meta=meta).compute())
print(i,j+k*remove,i+j+k*remove)
print('total number of bins before',np.sum(bins))
print('total number of bins after',np.sum(bins_after))
print(bins)
print(bins_after)
print(np.shape(del_grouped))
#np.save("without_noise_100000",del_grouped)
print ('Time taken for noise removal is {} minutes\n'.format(
                                                      (time.perf_counter()-start)/60))
print('noise removal done')


print('Number of particles after noise removal : ',len(del_grouped))

# print('Number of noise particles that were removed : ',len(noise))


#WITHOUT NOISE SECTION
# del_grouped=np.load("noise_100000.npy")
# print(np.shape(del_grouped))
# del del_grouped
# del_grouped=np.load("without_noise_100000.npy")
# print(np.shape(del_grouped))
# del del_grouped
#rng.shuffle(del_grouped, axis=0)
# def save_plot(X):
#     plt.figure(figsize=(6,6))
#     hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,vmin=0.01)
#     plt.xlabel('x in cms', fontsize=11)
#     plt.ylabel('y in cms', fontsize=11)
#     plt.title('xy_noise', pad=15)
#     cbar = plt.colorbar()
#     cbar.set_label('Counts in each pixel', labelpad=15, fontsize=11)
#     plt.subplots_adjust(left=0.15, bottom=0.15, right=0.82, top=0.85)
#     plt.savefig('xy_noise.png')
#     plt.close()
# save_plot(del_grouped)

# def save_plot(X):
#     plt.figure(figsize=(6,6))
#     hist,xedges,_=plt.hist(X[:,0],bins=100)
#     plt.title('x_noise', pad=15)
#     plt.xlabel('x in cms', fontsize=11)
#     plt.ylabel('y in cms', fontsize=11)
#     plt.subplots_adjust(left=0.15, bottom=0.15, right=0.82, top=0.85)
#     plt.savefig('x_noise.png')
#     plt.close()
#     return 
# save_plot(del_grouped)

# def save_plot(X):
#     plt.figure(figsize=(6,6))

#     hist,yedges,_=plt.hist(X[:,1],bins=100)
#     plt.title('y_noise',pad=15)
#     plt.xlabel('x in cms', fontsize=11)
#     plt.ylabel('y in cms', fontsize=11)
#     plt.subplots_adjust(left=0.15, bottom=0.15, right=0.82, top=0.85)
#     plt.savefig('y_noise.png')
#     plt.close()
#     return

# save_plot(del_grouped)


# def save_plot(X):
#     plt.figure(figsize=(6,6))
#     print('Number of particles in one batch : ',len(X))
#     hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,vmin=0.01)
#     plt.xlabel('x in cms', fontsize=11)
#     plt.ylabel('y in cms', fontsize=11)
#     plt.title('128*128*4_noise', pad=15)
#     cbar = plt.colorbar()
#     cbar.set_label('Counts in each pixel', labelpad=15, fontsize=11)
#     plt.subplots_adjust(left=0.15, bottom=0.15, right=0.82, top=0.85)
#     plt.savefig('128*128*4_noise.png')
#     plt.close()

# save_plot(del_grouped[:128*128*4,:])


# #NOISE SECTION


# def save_plot(X):
#     plt.figure(figsize=(6,6))
#     hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,vmin=0.01)
#     plt.title('xy_noise')
#     plt.colorbar()
#     plt.savefig('xy_noise.png')
#     plt.close()
# save_plot(noise)

# def save_plot(X):
#     plt.figure(figsize=(6,6))
#     print(np.shape(X[:,0]))
#     hist,xedges,_=plt.hist(X[:,0],bins=100)
#     plt.title('x_noise')
#     plt.savefig('x_noise.png')
#     plt.close()
#     return 
# save_plot(noise)

# def save_plot(X):
#     plt.figure(figsize=(6,6))

#     hist,yedges,_=plt.hist(X[:,1],bins=100)
#     plt.title('y_noise')
#     plt.savefig('y_noise.png')
#     plt.close()
#     return

# save_plot(noise)





# def save_plot(X):
#     plt.figure(figsize=(6,6))
#     print('Number of particles in one batch : ',len(X))
#     hist=plt.hist2d(X[:,0],X[:,1],bins=100,cmap=cmap,vmin=0.01)
#     plt.title('128*128*4 noise')
#     plt.colorbar()
#     plt.savefig('128*128*4_noise.png')
#     plt.close()

# save_plot(noise[:128*128*4,:])






























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



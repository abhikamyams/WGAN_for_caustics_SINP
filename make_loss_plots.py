import matplotlib.pyplot as plt
import dask.dataframe as dd
import numpy as np

#load FastSim and FullSim data
FastSim=np.load('/home/ubuntu/Abhikamya/Final/FastSim_phonons_total.npy',allow_pickle=True)
FullSim=dd.read_parquet('/home/ubuntu/Abhikamya/Original_root_files/Output1.parquet', 
columns=["Final_positionX", "Final_positionY","Deposited_energy", "Final_time"])

#take the same number of phonons from both data sets
FullSim=np.array(FullSim.head(FastSim.shape[0],npartitions=2))

#plot FullSim 1D histograms
plt.figure(figsize=(12,12))
plt.subplot(2,2,1,title='Final Position X',xlabel='X in m', ylabel='Counts in each bin')
plt.hist(FullSim[:,0],bins=100)

plt.subplot(2,2,2,title='Final Position Y',xlabel='Y in m', ylabel='Counts in each bin')
plt.hist(FullSim[:,1],bins=100)

plt.subplot(2,2,3,title='Deposited Energy',xlabel='Energy in eV', ylabel='Counts in each bin')
plt.hist(FullSim[:,2],bins=100)
plt.xscale('log')

plt.subplot(2,2,4,title='Final Time',xlabel='Time ns ', ylabel='Counts in each bin')
plt.hist(FullSim[:,3],bins=100)
plt.tight_layout()
plt.savefig('FullSim_1D.png')
plt.close('all')


#plot FastSim 1D histograms
plt.figure(figsize=(12,12))
plt.subplot(2,2,1,title='Final Position X',xlabel='X in m', ylabel='Counts in each bin')
plt.hist(FastSim[:,0],bins=100)

plt.subplot(2,2,2,title='Final Position Y',xlabel='Y in m', ylabel='Counts in each bin')
plt.hist(FastSim[:,1],bins=100)

plt.subplot(2,2,3,title='Deposited Energy',xlabel='Energy in eV', ylabel='Counts in each bin')
plt.hist(FastSim[:,2],bins=100)
plt.xscale('log')

plt.subplot(2,2,4,title='Final Time',xlabel='Time ns ', ylabel='Counts in each bin')
plt.hist(FastSim[:,3],bins=100)
plt.tight_layout()
plt.savefig('FastSim_1D.png')
plt.close('all')


#plot comparison histograms 
plt.figure(figsize=(12,12))
plt.subplot(2,2,1,title='Final Position X',xlabel='X in m', ylabel='Counts in each bin')
plt.hist(FullSim[:,0],bins=100,alpha=0.5,label='FullSim')
plt.hist(FastSim[:,0],bins=100,alpha=0.5,label='FastSim')

plt.subplot(2,2,2,title='Final Position Y',xlabel='Y in m', ylabel='Counts in each bin')
plt.hist(FullSim[:,1],bins=100,alpha=0.5,label='FullSim')
plt.hist(FastSim[:,1],bins=100,alpha=0.5,label='FastSim')

plt.subplot(2,2,3,title='Deposited Energy',xlabel='Energy in eV', ylabel='Counts in each bin')
plt.hist(FullSim[:,2],bins=100,alpha=0.5,label='FullSim')
plt.hist(FastSim[:,2],bins=100,alpha=0.5,label='FastSim')
plt.xscale('log')

plt.subplot(2,2,4,title='Final Time',xlabel='Time ns ', ylabel='Counts in each bin')
plt.hist(FullSim[:,3],bins=100,alpha=0.5,label='FullSim')
plt.hist(FastSim[:,3],bins=100,alpha=0.5,label='FastSim')
plt.tight_layout()
plt.legend()
plt.savefig('Comparision_histograms.png')
plt.close('all')


#energy comparision zoomed in on the discrete range
plt.plot(title='Deposited Energy',xlabel='Energy in eV', ylabel='Counts in each bin')
plt.hist(np.log(FullSim[:,2]),bins=1000,alpha=0.5,label='FullSim',range=(-22,-8))
plt.hist(np.log(FastSim[:,2]),bins=1000,alpha=0.5,label='FastSim',range=(-22,-8))

plt.yscale('log')
plt.legend()
plt.savefig('energy_comparison.png')

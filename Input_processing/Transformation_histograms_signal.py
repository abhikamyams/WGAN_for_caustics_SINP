
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import QuantileTransformer 


qtx = QuantileTransformer(output_distribution='normal', n_quantiles=1000000,subsample=20000000)
qty = QuantileTransformer(output_distribution='normal', n_quantiles=1000000,subsample=20000000)
qte = QuantileTransformer(output_distribution='normal', n_quantiles=1000000,subsample=20000000)
qtt = QuantileTransformer(output_distribution='normal', n_quantiles=1000000,subsample=20000000)

X=np.load('/home/ubuntu/Abhikamya/noise_removal/downconversion/signal_1.npy',allow_pickle=True)



plt.figure(figsize=(12,12))
plt.subplot(2,2,1,title='Final Position X',xlabel='X in m', ylabel='Counts in each bin')
plt.hist(X[:,0],bins=1000)

plt.subplot(2,2,2,title='Final Position Y',xlabel='Y in m', ylabel='Counts in each bin')
plt.hist(X[:,1],bins=1000)
plt.subplot(2,2,3,title='Deposited Energy',xlabel='Energy in eV', ylabel='Counts in each bin')
plt.hist(X[:,2],bins=1000)
plt.xscale('log')

plt.subplot(2,2,4,title='Final Time',xlabel='Time ns ', ylabel='Counts in each bin')
plt.hist(X[:,3],bins=1000)
plt.tight_layout()
plt.savefig('Original_histograms.png')
plt.close('all')

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


X[:,0]=(X[:,0] - minx)/(maxx-minx)
X[:,1]=(X[:,1] - miny)/(maxy-miny)
X[:,3]=(np.log(X[:,3])-logmint)/(logmaxt-logmint)
X[:,2]=(np.log(X[:,2])-logmine)/(logmaxe-logmine)

plt.figure(figsize=(12,12))



plt.subplot(2,2,1,title='Final Position X',xlabel='X in m', ylabel='Counts in each bin')
plt.hist(X[:,0],bins=1000)

plt.subplot(2,2,2,title='Final Position Y',xlabel='Y in m', ylabel='Counts in each bin')
plt.hist(X[:,1],bins=1000)
plt.subplot(2,2,3,title='Deposited Energy',xlabel='Energy in eV', ylabel='Counts in each bin')
plt.hist(X[:,2],bins=1000)


plt.subplot(2,2,4,title='Final Time',xlabel='Time ns ', ylabel='Counts in each bin')
plt.hist(X[:,3],bins=1000)
plt.tight_layout()
plt.savefig('Regularized.png')
plt.close('all')





X[:,0]=qtx.fit_transform(X[:,0].reshape(-1,1)).reshape(np.shape(X)[0])
X[:,1]=qty.fit_transform(X[:,1].reshape(-1,1)).reshape(np.shape(X)[0])
X[:,3]=qtt.fit_transform(X[:,3].reshape(-1,1)).reshape(np.shape(X)[0])
X[:,2]=qte.fit_transform(X[:,2].reshape(-1,1)).reshape(np.shape(X)[0])


plt.figure(figsize=(12,12))



plt.subplot(2,2,1,title='Final Position X',xlabel='X in m', ylabel='Counts in each bin')
plt.hist(X[:,0],bins=1000)

plt.subplot(2,2,2,title='Final Position Y',xlabel='Y in m', ylabel='Counts in each bin')
plt.hist(X[:,1],bins=1000)
plt.subplot(2,2,3,title='Deposited Energy',xlabel='Energy in eV', ylabel='Counts in each bin')
plt.hist(X[:,2],bins=1000)


plt.subplot(2,2,4,title='Final Time',xlabel='Time ns ', ylabel='Counts in each bin')
plt.hist(X[:,3],bins=1000)
plt.tight_layout()
plt.savefig('Quantile Transformation.png')
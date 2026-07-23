import numpy as np
import matplotlib.pyplot as plt
cmap = plt.cm.viridis.copy()
cmap.set_under('black')  

def save_plot2(X,Name,model):
    
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




    plt.suptitle(f'{Name} set {model}')
    plt.savefig(f'{Name}_{model}.png')
    plt.close()

#plot signal and noise
[maxx,maxy,minx,miny,mint,maxt,mine,maxe]=list(np.load('full_range.npy',allow_pickle=True))
clean =np.load('signal_1.npy',allow_pickle=True)
save_plot2(clean,'Signal','FullSim')
del clean
noise =np.load('noise_1.npy',allow_pickle=True)
save_plot2(noise,'Noise','FullSim')
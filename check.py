import numpy as np 
import matplotlib.pyplot as plt
results = np.load('results.npy')

print(np.shape(results))
w_dist=results[0,:]
kl=results[2,:]
epochs=[i for i in range (len(w_dist))]
gradient_penalty=results[3,:]
plt.plot(epochs,w_dist,label='w_dist')
plt.plot(epochs,gradient_penalty,label='gp')
# plt.xlabel('x in cms', fontsize=11)
plt.xlabel('epochs', fontsize=11)
# plt.yscale('log')
plt.title('w distance and gradient penalty')
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.85)
plt.grid(visible=True)
plt.legend()
plt.savefig('w_dist_and_gp.png')
plt.close('all')

# plt.plot(epochs,w_dist,label='w_dist')
# plt.plot(epochs,gradient_penalty,label='gp')
# plt.grid(visible=True)
# plt.subplots_adjust(left=0.15, bottom=0.15, right=0.82, top=0.85)
# plt.legend()
# plt.savefig('w_dist_and_gp.png')
# plt.close('all')

plt.plot(epochs,kl,label='kl divergence')
plt.grid(visible=True)
plt.xlabel('epochs', fontsize=11)
plt.ylabel('kl divergence')
plt.yscale('log')
plt.subplots_adjust(left=0.2, bottom=0.15, right=0.8, top=0.85)
plt.legend()
plt.savefig('Kl_divergence.png')
plt.close('all')

plt.plot(epochs,-w_dist+gradient_penalty,label='d_loss')
plt.xlabel('epochs', fontsize=11)
plt.plot(epochs,-w_dist,label='-wdist')
plt.plot(epochs,gradient_penalty,label='gradient penalty')
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.85)
plt.grid(visible=True)

plt.legend()
plt.savefig('d_loss.png')

# plt.title(f'Histograms for caustics 1/10, after epoch {epoch}',pad=15)
# plt.xlabel('x in cms', fontsize=11)
# plt.ylabel('y in cms', fontsize=11)
#     cbar = plt.colorbar()
#     cbar.set_label('Counts in each pixel', labelpad=15, fontsize=11)
#     plt.subplots_adjust(left=0.15, bottom=0.15, right=0.82, top=0.85)
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
'''
parameters() # bottom level, no names
named_parameters() # bottom level, includes names
named_modules() # recursive
named_children() # top-level
'''

def get_norms(model):
    with torch.no_grad():
        norms = []
        theta = [(p.data,k) for k,p in model.named_parameters()]
        for param,k in theta:
            if param.dim()==1:
                continue
            print(f"{k} -- {param.shape}")
            for f in param:
                norms.append(torch.norm(f).item())
    return np.asarray(norms)
        
def compare_norms(norms, epochs=[30,60,90,119]):
    for i in range(len(epochs)):
        diff = np.divide(norms[i],norms[0])
        if i==1:
            print(diff.shape)
        mean,std = np.mean(diff), np.std(diff)
        print(f"{mean=}     {std=}")
    return    
    
# Double check results in-place
def _check_direction(direction, theta_star, ignore=False):
    # Loop over parameter tensors
    for i,p in enumerate(direction):
        # Loop over filters/FC neurons
        for d,theta in zip(p,theta_star[i][0]):
            if ignore and p.dim()==1:
                assert torch.norm(p)==0.0
            else:
                assert torch.allclose(torch.norm(d),torch.norm(theta))

# Returns one filter-normed random direction in parameter space
def get_direction(model, theta_star, ignore=False):
    
    # initalize random direction by sampling from Gaussian
    direction = [torch.randn(v.shape, device=torch.device('cuda:0')) for v,_ in theta_star]
    
    # Iterate over parameters in d
    for i,p in enumerate(direction):
        # Wipes elements associated with batchnorm and bias params
        if ignore and p.dim()==1:
            print(theta_star[i][1])
            p.mul_(0.0)
        else:
            # Iterate over filters in one layer and normalize
            for d,theta in zip(p,theta_star[i][0]):
                theta_norm = torch.norm(theta)
                d_norm = torch.norm(d)
                d.mul_(theta_norm / (d_norm))
                assert torch.allclose(torch.norm(d), theta_norm)        
        print()
    
    _check_direction(direction, theta_star, ignore=ignore)
    return direction

def perturb_theta(model, theta_star, dir1, dir2, pos):
    # Move in parameter space
    theta = []
    for i,(p_star,_) in enumerate(theta_star):
        theta.append(p_star + pos[0]*dir1[i] + pos[1]*dir2[i])
    # Update model
    for i,(k,p) in enumerate(model.named_parameters()):
        p.data = theta[i]
    return

def reset_theta(model, theta_star):
    for i,(k,p) in enumerate(model.named_parameters()):
        p.data = theta_star[i][0]
    return


def plot_landscape(model,plot_type='trisurf', indx=-1):
    d = model.metrics['visual'][indx]
    alpha_grid,beta_grid = torch.meshgrid(d['alphas'],d['betas'])
    losses = np.nan_to_num(d['losses'], nan=np.Inf)

    vmin,vmax = 0.0,60.0
    losses = np.clip(losses, a_min=vmin, a_max=vmax)
    
    print(f"{vmin=}")
    print(f"{vmax=}")
    
    fig = plt.figure(figsize=(12,4))
    # Loss landscape
    if plot_type=='contour':
        ax = fig.add_subplot(122)
        surf = ax.contour(alpha_grid, beta_grid, losses, levels=np.linspace(vmin,vmax,21), cmap='magma')        
    elif plot_type=='trisurf':
        ax = fig.add_subplot(122,projection='3d')
        ax.set_zlim(bottom=vmin, top=vmax)
        surf = ax.plot_trisurf(alpha_grid.flatten(), beta_grid.flatten(), losses.flatten(),
                               norm=colors.Normalize(vmin,vmax), cmap='magma')
        ax.set_zlabel(f"{d['split']} loss")    
    ax.set_title("Loss landscape")
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    # Histogram of loss values collected
    ax_hist = fig.add_subplot(121)
    ax_hist.set_title("Histogram of losses")
    ax_hist.hist(losses.flatten(), bins=20)
    ax_hist.set_xlabel('Loss value')
    ax_hist.set_ylabel('Frequency')
    
    fig.subplots_adjust(wspace=0.5)    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    return
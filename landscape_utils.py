import numpy as np
import torch

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
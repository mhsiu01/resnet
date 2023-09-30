import pdb
from tqdm import tqdm
from time import time, sleep

import numpy as np
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp

# Fix RNG
import random
seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def main(args):
    # model
    from model_clean import ResNet_cifar, get_optimizer
    model = torch.load(f'./models/{args.model}.pth')
    model.to(memory_format=torch.channels_last)
    model = model.to('cuda:0')
    model.eval()
    conf = model.conf
#     conf['USE_FLOAT16'] = False

    # training set
    import dataloading
    from dataloading import write_datasets, get_loader
    train_loader = get_loader(conf['MEAN'], conf['STD'], split=args.split, BS=conf['BS'], SCALER=4*conf['SCALER'], USE_FLOAT16=False, visualize=True) #conf['USE_FLOAT16']
    print(f"Batch size is {conf['BS']}x{conf['SCALER']} = {conf['BS']*conf['SCALER']}.")

    # Print filter norms
    from landscape_utils import get_direction, perturb_theta, reset_theta, compare_norms
    compare_norms(model.metrics['norms'])

    # Generate two random directions in parameter space
    conf['ignore'] = True
    theta_star = [(p.data,k) for k,p in model.named_parameters()]
    dir1 = get_direction(model, theta_star, conf['ignore'])
    dir2 = get_direction(model, theta_star, conf['ignore'])

    # Meshgrid over alpha and beta. 
    R = args.R
    steps = args.steps
    alphas = torch.linspace(-R,R, steps=steps)
    betas = torch.linspace(-R,R, steps=steps)
    alpha_grid,beta_grid = torch.meshgrid(alphas,betas)
    coords = torch.stack((alpha_grid, beta_grid), dim=2)

    # Evaluate
    from utils_clean import evaluate
    losses = []
    for pos in tqdm(torch.reshape(coords, (-1,2))):
        perturb_theta(model, theta_star, dir1, dir2, pos)
        loss,acc = evaluate(model, train_loader, USE_FLOAT16=False) #conf['USE_FLOAT16']
        losses.append(loss)
    losses = np.asarray(losses).reshape(tuple(alpha_grid.shape))
    print(losses.shape)

    # Reset model to original parameter values
    reset_theta(model, theta_star)
    for i,(k,p) in enumerate(model.named_parameters()):
        assert (p.data == theta_star[i][0]).all()

    # Save relevant variables to model
    vis_dict = {
        'R':R,
        'steps':steps,
        'split':args.split, 
        'alphas':alphas,
        'betas':betas,
        'losses':losses,
        'dirs':(dir1,dir2),
        'ignore':conf['ignore'],
    }
    if 'visual' not in model.metrics.keys():
        model.metrics['visual'] = [vis_dict]
    else:
        model.metrics['visual'].append(vis_dict)
    torch.save(model,f'./models/{args.model}.pth')
    print("Loss evaluation saved.")

    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Model to visualize.", required=True) # eg. "cifar100_n9_"
    parser.add_argument('--R', default=1.0, type=float, help="Range of values to visualize, ie. alpha,beta in [-R,+R].", required=False)
    parser.add_argument('--steps', default=11, type=int, help="Number of points to evaluate along each direction.", required=False)
    parser.add_argument('--split', default='train', type=str, help="Data split to use for evaluation.", required=False)
    args = parser.parse_args()
    main(args)

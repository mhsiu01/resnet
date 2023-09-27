# Imports
import pdb
from tqdm import tqdm
from time import time, sleep
import numpy as np
import torch
import torch.nn.functional as F
# Fix random seed
import random
seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def main(conf):
    # ffcv dataloading
    import dataloading
    from dataloading import write_datasets, get_loader
    MEAN,STD = write_datasets(num_classes=conf['num_classes'], lengths=conf['lengths'], write=conf['write_beton'])
    conf['MEAN'] = MEAN
    conf['STD'] = STD
    train_loader = get_loader(conf['MEAN'], conf['STD'],
                              split='train', num_classes=conf['num_classes'],
                              BS=conf['BS'], SCALER=conf['SCALER'])
    val_loader = get_loader(conf['MEAN'], conf['STD'],
                            split='val', num_classes=conf['num_classes'],
                            BS=conf['BS'], SCALER=conf['SCALER'])
    print(f"Batch size is {conf['BS']}x{conf['SCALER']} = {conf['BS']*conf['SCALER']}.")

    # Metrics
    from utils_clean import evaluate
    metrics = {
        'train_losses':[],
        'val_losses':[],
        'val_accs':[],
        'times':[],
        'norms':None,
    }
    
    # Model / optimizer
    from model_clean import ResNet_cifar, get_optimizer
    import torch.cuda.amp as amp
    model = ResNet_cifar(conf=conf, metrics=metrics)
    model.to(memory_format=torch.channels_last)
    model = model.to('cuda:0')
    model.train()
    print(f"total params:{sum(p.numel() for p in model.parameters())}")
    optimizer, scheduler = get_optimizer(model, SCALER=conf['SCALER'])
    scaler = amp.GradScaler()

    # Train loop
    from landscape_utils import get_norms
    for epoch in tqdm(range(conf['num_epochs'])):
        print(f"Epoch #{epoch} has lr={scheduler.get_last_lr()}")
        total_loss = 0.0
        t_start = time()
        model.train()
        for x,y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            # Forward pass
            with amp.autocast(enabled=conf['USE_FLOAT16']):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            # Gradient descent with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # Formula: avg_loss_per_example * num_examples = minibatch_cumulative_loss
            total_loss += (loss.item() * x.shape[0]) 
        scheduler.step()

        # Average training loss
        train_loss = total_loss / len(train_loader.indices)
        metrics['train_losses'].append(train_loss)

        # Average validation loss and prediction accuracy
        val_loss, val_acc = evaluate(model,val_loader)
        metrics['val_losses'].append(val_loss)
        metrics['val_accs'].append(val_acc)
        print("val_acc = {}".format(round(val_acc,2)))

        # Track time per epoch
        t_finish = time()
        metrics['times'].append(t_finish - t_start)
        print("t = {}".format(round(t_finish - t_start,3)))
        
        # Store weight norms
        if epoch%1==0 and conf['norms']:
            norms = get_norms(model)[None,:]
            if metrics['norms'] is None:
                metrics['norms'] = norms
            else:
                metrics['norms'] = np.concatenate((metrics['norms'], norms), axis=0)
                print(metrics['norms'].shape)
        print()

    # Plot
    from utils_clean import plot_metrics
    # f = f"cifar100-{conf['lengths']}-BS{conf['BS']}x{conf['SCALER']}"
    plot_metrics(metrics, save=conf['plot_save'])
    print(f"Final train loss: {metrics['train_losses'][-1]}")
    print(f"Final val accuracy: {metrics['val_accs'][-1]}")
    print(f"Average epoch time: {sum(metrics['times']) / len(metrics['times'])}")
    
    # Save model
    if conf['model_save']!="":
        f = f"./models/{conf['model_save']}.pth"
        torch.save(model,f)
        print(f"Saved model as \'{f}\'.")
    else:
        print("Not saving model.")
    print("Training done.")
    
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=3, type=int, help="Number of residual blocks per layer.", required=False)
    parser.add_argument('--k', default=1, type=int, help="Widening factor for Wide Resnets.", required=False)
    parser.add_argument('--NC', default=100, type=int, help="Number of classes to classify into. 10 or 100 for CIFAR10/100 respectively.", required=False)
    parser.add_argument('--no_skip', default=0, type=int, help="Disable skip connections.", required=False)
    parser.add_argument('--BS', default=128, type=int, help="Batch size.", required=False)
    parser.add_argument('--SCALER', default=4, type=int, help="Scales up batch size.", required=False)
    parser.add_argument('--NE', default=120, type=int, help="Number of epochs.", required=False)
    parser.add_argument('--lengths', default=[40000,10000], nargs="+", help="Number of examples in train and val splits.", required=False)
    parser.add_argument('--USE_FLOAT16', default=1, type=int, help="Reduced precision training", required=False)
    parser.add_argument('--plot_save', default="", type=str, help="File name of metrics plot. Leave blank to save nothing.", required=False)
    parser.add_argument('--model_save', default="", type=str, help="Save model checkpoint. Leave blank to save nothing.", required=False)
    parser.add_argument('--beton', default=0, type=int, help="Write new .beton files. 1 for new files, 0 to use existing.", required=False)
    parser.add_argument('--norms', default=0, type=int, help="Optionally store parameter norms per filter (for visualizing loss landscape). Ignores bias and batchnorm. Argument is int (later converted to boolean).", required=False)
    args = parser.parse_args()
    
    # Hyperparams
    conf = {
        # Model
        'n':args.n,
        'k':args.k,
        'num_classes':args.NC,
        'no_skip':bool(args.no_skip),
        # Data / training behavior
        'BS':args.BS,
        'SCALER':args.SCALER,
        'num_epochs':args.NE,
        'lengths':[int(x) for x in args.lengths],
        'USE_FLOAT16':bool(args.USE_FLOAT16),
        # Save stuff
        'plot_save':args.plot_save,
        'model_save':args.model_save,
        'write_beton':bool(args.beton),
        'norms':bool(args.norms),
    }
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(conf)
    
    main(conf)
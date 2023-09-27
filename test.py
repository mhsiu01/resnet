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
    test_loader = get_loader(conf['MEAN'], conf['STD'],
                             split='test', num_classes=conf['num_classes'],
                             BS=conf['BS'], SCALER=conf['SCALER'],
                             DEVICE=torch.device('cuda'), USE_FLOAT16=conf['USE_FLOAT16'])
    print(f"Batch size is {conf['BS']}x{conf['SCALER']} = {conf['BS']*conf['SCALER']}.")

    # Metrics
    from utils_clean import evaluate
    metrics = {
        'test_losses':[],
        'test_accs':[],
    }
    
    # Load model
    import torch.cuda.amp as amp
    model = torch.load(f"./models/{conf['checkpoint']}.pth")
    model.to(device='cpu', memory_format=torch.channels_last)
    model = model.to('cuda')
    print(next(model.parameters()).device)
    model.eval()
    print(f"total params:{sum(p.numel() for p in model.parameters())}")
    scaler = amp.GradScaler()

    # Evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader, conf['USE_FLOAT16'])
    print("test_acc = {}".format(round(test_acc,2)))
    
    return
    
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help="Name of saved model file to test accuracy on.", required=True)
    parser.add_argument('--NC', default=100, type=int, help="Number of classes to classify into. 10 or 100 for CIFAR10/100 respectively.", required=False)
    parser.add_argument('--lengths', default=[40000,10000], nargs="+", help="Number of examples in train and val splits.", required=False)
    parser.add_argument('--no_skip', default=0, type=int, help="Disable skip connections.", required=False)
    parser.add_argument('--BS', default=128, type=int, help="Batch size.", required=False)
    parser.add_argument('--SCALER', default=4, type=int, help="Scales up batch size.", required=False)
    parser.add_argument('--USE_FLOAT16', default=1, type=int, help="Reduced precision training", required=False)    
    parser.add_argument('--beton', default=0, type=int, help="Write new .beton files. 1 for new files, 0 to use existing.", required=False)
    args = parser.parse_args()
    
    # Hyperparams
    conf = {
        # Model
        'checkpoint':args.checkpoint,
        'num_classes':args.NC,
        'no_skip':bool(args.no_skip),
        # Data / training behavior
        'lengths':args.lengths,
        'BS':args.BS,
        'SCALER':args.SCALER,
        'USE_FLOAT16':bool(args.USE_FLOAT16),
        'write_beton':bool(args.beton)
    }
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(conf)
    
    main(conf)
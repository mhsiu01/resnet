import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp

def evaluate(model, loader, USE_FLOAT16=True):
    preds = []
    targets = []
    with torch.no_grad():
        model.eval()
        
        # Forward pass
        total_loss = 0.0
        for x,y in loader:
            with amp.autocast(enabled=USE_FLOAT16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            total_loss += (loss.item() * x.shape[0])
            preds = preds + list(torch.argmax(logits, dim=1))
            targets = targets + list(y.detach().clone().cpu())
            
        # Accuracy
        avg_loss = total_loss / len(loader.indices)
        num_correct = sum(1 for pred,target in zip(preds, targets) if pred == target)
        accuracy = 100*(num_correct / len(targets))
        
    return avg_loss, accuracy



def plot_metrics(metrics, save=""):
    # data should be list of (python list, string label)
    graphs = [
            # losses
            {
                'data':[
                    (metrics['train_losses'],'train'),
                    (metrics['val_losses'],'val')
                ],
                'x_label':"Epoch",
                'y_label':"Loss",
                'title': "Train/val losses"
            },
            # accuracies
            {
                'data':[(metrics['val_accs'],None)],
                'x_label':"Epoch",
                'y_label':"Accs",
                'title': "Val accuracy"
            },
            # epoch times
            {
                'data':[(metrics['times'],None)],
                'x_label':"Epoch",
                'y_label':"Seconds",
                'title': f"Time per epoch"
            }
    ]
    plt.rcParams.update({'font.size': 12})
    G = len(graphs)
    fig = plt.figure(figsize=(10*G,6))
    axes = [fig.add_subplot(1,G,i+1) for i in range(G)]
    
    for i,ax in enumerate(axes):
        g = graphs[i]
        for data,label in g['data']:
            ax.plot(data, label=label)
        ax.set_xlabel(g['x_label'])
        ax.set_ylabel(g['y_label'])
        ax.set_title(g['title'])
        ax.legend()
    if save=="":
        print(f"Not saving plot.")
    else:
        plt.savefig(f'./plots/{save}.pdf', dpi=150., pad_inches=1.0)
        print(f"Plot saved as \'{save}\'.")
        
    plt.show()

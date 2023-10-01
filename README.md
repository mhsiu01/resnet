# resnet
## Training
Implements ResNet architecture and trains ResNet-18 on CIFAR-10/100.

Performance is roughly the same as in the Wide Residual Networks paper (https://arxiv.org/abs/1605.07146),
specifically 94.53% for CIFAR-10 and 78.45% for CIFAR-100 (test accuracy, k=8 for wider layers).
To accelerate training, I used larger batch sizes and scaled learning rates as proposed in the "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" paper (https://arxiv.org/abs/1706.02677).

Roughly 2-2.5x training speedup was observed when using ffcv dataloader over PyTorch dataloader.

Note 1: The plots show the final training run using both training and validation sets together,
so the validation accuracy/loss curves are meaningless.

Note 2: Run train.py from command line. To run on Google colab, use colab_wrapper.ipynb. This will clone the repo. If running for the first time or changing data splits, change --beton flag to 1. Subsequent runs can use --beton 0 to save time.

Note 3: To train on DSMLP with ffcv, one needs to create a custom Docker container since package installs are restricted. See my 'ffcv_docker' repo.

## Loss landscape
I also implemented "filter normalization" from "Visualizing the Loss Landscape of Neural Nets"(https://arxiv.org/abs/1712.09913). The trained model seems to be in a fairly nice bowl-shaped minima:
![image](https://github.com/mhsiu01/resnet/assets/78574718/b14c57ed-0eb2-473a-bd74-84570e6e906a)




# resnet
Implements ResNet architecture and trains ResNet-18 on CIFAR-10/100.

Performance is roughly the same as in the Wide Residual Networks paper (https://arxiv.org/abs/1605.07146),
specifically around 94.53% for CIFAR-10 and 78.45% for CIFAR-100.
To accelerate training, we make use of larger batch sizes and scaled learning rates as proposed in the "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" paper (https://arxiv.org/abs/1706.02677).

Roughly 2-2.5x training speedup was observed when using ffcv dataloader over PyTorch dataloader.

Note 1: The plots show the final training run using both training and validation sets together,
so the validation accuracy/loss curves are meaningless.

Note 2: Run train.py from command line. To run on Google colab, use colab_wrapper.ipynb. This will clone the repo. If running for the first time or changing data splits, change --beton flag to 1. Subsequent runs can use --beton 0 to save time.

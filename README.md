# resnet
Implements ResNet architecture and trains ResNet-18 on CIFAR-10/100.
Performance is roughly the same as in the Wide Residual Networks paper (https://arxiv.org/abs/1605.07146),
specifically around 94.53% for CIFAR-10 and 78.45% for CIFAR-100.

Roughly 2x training speedup was observed when using ffcv dataloader over PyTorch dataloader.

Note 1: The plots show the final training run using both training and validation sets together,
so the validation accuracy/loss curves are meaningless.

Note 2: When training, create './data' and './data/ffcv' as directories to hold original dataset and the ffcv .beton files, respectively.

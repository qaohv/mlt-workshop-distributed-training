"""
Distributed training + mixed precision usage simple example.
"""
import argparse
import time
import os

import numpy as np
import torch
import torchvision.transforms as transforms

from apex import amp
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.models import resnet50
from tqdm import tqdm

from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.distributed import DistributedSampler


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer_, start_lr, finish_lr, warmup_epochs_num, last_epoch=-1):
        self.start_lr = start_lr
        self.finish_lr = finish_lr
        self.warmup_epochs = warmup_epochs_num

        lr_step = (finish_lr - start_lr) / self.warmup_epochs
        self.lrs = [self.start_lr + lr_step * i for i in range(self.warmup_epochs + 1)]

        super().__init__(optimizer_, last_epoch)

    def get_lr(self):
        print(f"last epoch: {self.last_epoch}")
        lr = self.lrs[self.last_epoch] if self.last_epoch + 1 < self.warmup_epochs else self.finish_lr
        print(f"lr is : {lr})")
        return [lr]


class Resnet50Classifier(torch.nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        self.backbone = torch.nn.Sequential(*list(resnet50(pretrained=True).children())[:-1])
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=512, bias=False),
            torch.nn.Dropout(p=.5),
            torch.nn.Linear(in_features=512, out_features=num_classes, bias=False),
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out.view(out.size()[0], -1))
        return out


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    # important flags!
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=None,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train NN.')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size to train NN.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate to train NN.')
    parser.add_argument('--warmup-epochs', type=int, default=0, help='Warmup epochs number')
    parser.add_argument('--warmup-start-lr', type=float, default=1e-2, help='Warmup low lr bound.')
    parser.add_argument('--use-mixed-precision', type=str, default="O0", choices=["O0", "O1"],
                        help='Disable or enable mixed precision training.')

    args = parser.parse_args()
    # Dist training
    set_random_seeds(random_seed=0)

    dist_training = int(os.environ.get("WORLD_SIZE", 1)) > 1
    # Dist training
    if dist_training:
        torch.distributed.init_process_group(backend="nccl")

    # Dist training
    device = torch.device(f"cuda:{args.local_rank or 0}")

    net = Resnet50Classifier(num_classes=100).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.warmup_start_lr if args.warmup_epochs > 0 else args.lr,
                                momentum=0.9, weight_decay=1e-4)

    net, optimizer = amp.initialize(net, optimizer, opt_level=args.use_mixed_precision)

    # Explain warmup
    scheduler = None
    if args.warmup_epochs > 0:
        scheduler = WarmupScheduler(optimizer, args.warmup_start_lr, args.lr, args.warmup_epochs)

    # Dist training
    if dist_training:
        net = torch.nn.parallel.DistributedDataParallel(net,
                                                        device_ids=[args.local_rank],
                                                        output_device=args.local_rank)

    criterion = torch.nn.CrossEntropyLoss()

    transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])

    train_dataset = CIFAR100(root='./', download=True, train=True, transform=transforms)
    # Dist training
    if dist_training:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      sampler=DistributedSampler(dataset=train_dataset), num_workers=2)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    epoch_train_loss = []
    for epoch in range(args.epochs):
        start = time.time()
        for images, labels in tqdm(train_dataloader):
            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)
            predictions = net(images)

            loss = criterion(predictions, labels)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()
            epoch_train_loss.append(loss.item())

        finish = time.time()

        if scheduler is not None:
            scheduler.step(epoch)

        print(f"Epoch: {epoch}, train loss: {np.mean(epoch_train_loss):.5f}, epoch time: {finish - start}")

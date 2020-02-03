__author__ = "Aditya Singh"
__version__ = "0.1"

import torch
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from models.resnet import resnet18
from utils import mixup_criterion, mixup_data, compute_calibration_metrics, save_checkpoint
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
times = 1

for time in range(times):
    TIME_NOW = str(datetime.now().strftime('%Y-%m-%d--%H-%M'))

    for alpha in alphas:
        ### Hyperparameters
        epochs = 200
        data_collection_milestones = [0, 25, 50, 75, 100, 125, 150, 175, epochs-1]
        learning_rate = 0.1     # This value seems to be working for pytorch
        learning_rate_milestones = [60, 120, 160]
        learning_gamma = 0.2
        nesterov = True
        momentum = 0.9    # exact value NOT specified in the paper
        weight_decay = 0.0005
        NUM_BINS = 100

        ### Output dirs for the experiments
        checkpoint = os.path.join('checkpoints/resnet18/fmnist','alpha:{}'.format(alpha), TIME_NOW)
        log_path = os.path.join(checkpoint, 'logs')
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
            os.makedirs(log_path)

        writer = SummaryWriter(log_path)

        ### Training Data

        ##### CIFAR100 specific values
        mean = (0.2868,)
        std_dev = (0.3524,)

        transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std_dev)
            ])

        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std_dev)
        ])
        train_dataset = FashionMNIST(root='./data/', train=True, download=True, transform=transform_train)
        test_dataset = FashionMNIST(root='./data/', train=False, transform=transform_test, download=True)

        ### Loading the model
        # vgg = torchvision.models.vgg16_bn(pretrained=False, num_classes=100)
        net = resnet18(in_channels=1, num_classes=10)
        net = net.to(device)

        ### Optimiser
        optimiser = SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        scheduler = lr_scheduler.MultiStepLR(optimiser, milestones=learning_rate_milestones, gamma=learning_gamma)

        ### The loss function
        criterion = torch.nn.CrossEntropyLoss()

        ### The paper does not mention the batch size if used and the x and y labels are inconsitent
        batch_size = 128
        train_loader = DataLoader(train_dataset, shuffle=True, num_workers=2, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, shuffle=True, num_workers=2, batch_size=batch_size)

        accuracies = []
        means_of_winning_score = []
        for epoch in range(epochs):
            net.train()
            progress = tqdm(enumerate(train_loader), desc="Epoch: {}".format(epoch), total=len(train_loader))
            for iter, data in progress:
                images, labels = data[0], data[1]
                inputs, targets_a, targets_b, lam = mixup_data(images, labels, alpha=alpha)
                optimiser.zero_grad()
                outputs = net(inputs.to(device))
                inputs, targets_a, targets_b = inputs.to(device), targets_a.to(device), targets_b.to(device)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                loss.backward()
                optimiser.step()
                progress.update(1)
            scheduler.step()

            if epoch in data_collection_milestones:
                ece, oe, acc, conf, A, counts = compute_calibration_metrics(num_bins=NUM_BINS, net=net, loader=test_loader)
                save_checkpoint(path=checkpoint, net=net, acc=A, ece=ece, oe=oe, epoch=epoch, alpha=alpha)
                print('Accuracy: {}'.format(A))
                #plot_scatterplot(np.asarray(conf), np.asarray(acc), x_label='confidence', y_label='accuracy',
                #                 title='CalibrationPlot-{}'.format(epoch), checkpoint=checkpoint, weights=counts)
                writer.add_scalar('ECE:alpha:{}'.format(alpha), ece, global_step=epoch+1)
                writer.add_scalar('OE:alpha:{}.format(alpha)', oe, global_step=epoch + 1)
                writer.add_scalar('Accuracy:alpha:{}'.format(alpha), A, global_step=epoch + 1)



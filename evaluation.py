__author__ = "Aditya Singh"
__version__ = "0.1"

import yaml
import glob
import torch
import argparse
from datetime import datetime
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, STL10, FashionMNIST, MNIST
from utils import load_checkpoint
import numpy as np
from utils import plot_barplot, plot_scatterplot, plot_distplot
import pandas as pd
from utils import compute_calibration_metrics
import torch.distributions as tdist
import torch.utils.data as data_utils

TIME_NOW = str(datetime.now().strftime('%Y-%m-%d--%H-%M'))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def main(args):
    with open(args.config, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    NUMBER_OF_TRIALS = config['number_of_trials']
    parent_path = os.path.join(config['destination'], config['dataset'], config['network'], TIME_NOW)
    for epoch in config['epochs']:
        os.makedirs(os.path.join(parent_path, str(epoch)), exist_ok=True)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std_dev'])
    ])
    title = 'Confidence on Test Images'
    if config['dataset'] == 'cifar100':
        test_dataset = CIFAR100(root='./data/', train=False, transform=transform_test, download=True)
        test_loader = DataLoader(test_dataset, shuffle=False, num_workers=2, batch_size=128)
    if config['dataset'] == 'stl_10':
        test_dataset = STL10(root='./data/',split='test', transform=transform_test, download=True)
        test_loader = DataLoader(test_dataset, shuffle=False, num_workers=2, batch_size=128)
    if config['dataset'] == 'fmnist':
        test_dataset = FashionMNIST(root='./data/', train=False, transform=transform_test, download=True)
        test_loader = DataLoader(test_dataset, shuffle=False, num_workers=2, batch_size=128)
    if config['dataset'] == 'out-of-distribution':
        test_dataset = MNIST(root='./data/', train=False, transform=transform_test, download=True)
        test_loader = DataLoader(test_dataset, shuffle=False, num_workers=2, batch_size=128)
        title='Confidence on Out-of-Distribution Images'
        config['dataset'] = 'fmnist'
    if config['dataset'] == 'noise':
        normal = tdist.Normal(config['mean'][0], config['std_dev'][0])
        test_images = normal.sample((1024, 1, 28, 28))
        test_targets = torch.ones([1024], dtype=torch.long)
        test_dataset = data_utils.TensorDataset(test_images, test_targets)
        test_loader = data_utils.DataLoader(test_dataset, batch_size=128, shuffle=False)
        config['dataset'] = 'fmnist'
        title = 'Confidence on Gaussian Noise Images'


    base_path = os.path.join('checkpoints', config['network'], config['dataset'])
    results = pd.DataFrame(columns=['alpha', 'exp_id', 'acc', 'ece', 'oe', 'epoch', 'bin_conf', 'bin_acc', 'bin_count'])
    for alpha in config['alphas']:
        base_alpha_path = os.path.join(base_path, 'alpha:{}'.format(alpha))
        experiments = glob.glob(base_alpha_path+'/*')
        for epoch in config['epochs']:
            for idx, experiment in enumerate(experiments):
                model_path = os.path.join(experiment, 'ckpt-{}.t7'.format(epoch))
                net, _, details = load_checkpoint(filename=model_path)
                net = net.to(device)
                ece, oe, bin_acc, bin_conf, acc, bin_count = compute_calibration_metrics(num_bins=100, net=net, loader=test_loader)
                results = results.append({'alpha':alpha, 'exp_id': idx, 'acc': acc, 'ece': float(ece)
                                   , 'oe': float(oe), 'epoch': epoch, 'bin_conf': bin_conf, 'bin_acc': bin_acc, 'bin_count':bin_count}
                                         , ignore_index=True)

    results_df = results[results['epoch'] == 199]
    acc_df = results_df[['alpha', 'acc']]
    plot_barplot(acc_df, x='alpha', y='acc', title='Accuracy for different Alphas', xlabel='Alpha', ylabel='Accuracy', file=parent_path+'/accuracyValpha.png', view=False)
    ece_df = results_df[['alpha', 'ece']]
    plot_barplot(ece_df, x='alpha', y='ece', title='ECE for different Alphas', xlabel='Alpha', ylabel='Error', file=parent_path+'/eceValpha.png', view=False)
    oe_df = results_df[['alpha', 'oe']]
    plot_barplot(oe_df, x='alpha', y='oe', title='OE for different Alphas', xlabel='Alpha', ylabel='Error',
                 file=parent_path + '/oeValpha.png', view=False)


    for epoch in config['epochs']:
        epoch_results_df = results[results['epoch']==epoch]
        conf_dist = {}
        for alpha in config['alphas']:
            alpha_data = epoch_results_df[epoch_results_df['alpha'] == alpha]
            bin_acc = None
            bin_conf = None
            bin_count = None
            dist_conf = []
            for index, row in alpha_data.iterrows():
                if bin_count is None:
                    bin_acc, bin_conf, bin_count = row['bin_acc'], row['bin_conf'], row['bin_count']
                else:
                    bin_acc = [a+b for (a, b) in zip(bin_acc, row['bin_acc'])]
                    bin_conf = [a+b for (a, b) in zip(bin_conf, row['bin_conf'])]
                    bin_count = [a+b for (a, b) in zip(bin_count, row['bin_count'])]
            bin_acc = [a/NUMBER_OF_TRIALS for a in bin_acc]
            bin_conf = [a/NUMBER_OF_TRIALS for a in bin_conf]
            bin_count = [a/NUMBER_OF_TRIALS for a in bin_count]
            acc, conf, count = [], [], []
            for (i, j, k) in zip(bin_acc, bin_conf, bin_count):
                if k >= 1:
                    acc.append(i)
                    conf.append(j)
                    count.append(k)
                    for nos in range(int(k)):
                        dist_conf.append(j)
            conf_dist[alpha] = dist_conf
            plot_scatterplot(x=np.asarray(conf), y=np.asarray(acc), weights=np.asarray(count), title=None,
                                xlabel='Avg. bin confidence', ylabel='Avg. bin accuracy',file=parent_path+'/{}/scatterplot_{}.png'.format(epoch, alpha))

        plot_distplot(conf_dist, config['alphas'], title, ylabel=None, xlabel='Confidence', file=parent_path+'/{}/distplot.png'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation script to generate figures post training')
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()
    main(args)

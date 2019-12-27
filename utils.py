import numpy as np
import torch
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns
import os


def compute_calibration_metrics(num_bins=100, net=None, loader=None, device='cuda'):
    """
    Computes the calibration metrics ECE and OE along with the acc and conf values
    :param num_bins: Taken from email correspondence and 100 is used
    :param net: trained network
    :param loader: dataloader for the dataset
    :param device: cuda or cpu
    :return: ECE, OE, acc, conf
    """
    acc_counts = [0 for _ in range(num_bins+1)]
    conf_counts = [0 for _ in range(num_bins+1)]
    overall_conf = []
    n = float(len(loader.dataset))
    counts = [0 for i in range(num_bins+1)]
    net.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confs, preds = probabilities.max(1)
            for (conf, pred, label) in zip(confs, preds, labels):
                bin_index = int(((conf * 100) // (100/num_bins)).cpu())
                try:
                    if pred == label:
                        acc_counts[bin_index] += 1.0
                    conf_counts[bin_index] += conf.cpu()
                    counts[bin_index] += 1.0
                    overall_conf.append(conf.cpu())
                except:
                    print(bin_index, conf)
                    raise AssertionError('Bin index out of range!')


    avg_acc = [0 if count == 0 else acc_count / count for acc_count, count in zip(acc_counts, counts)]
    avg_conf = [0 if count == 0 else conf_count / count for conf_count, count in zip(conf_counts, counts)]
    ECE, OE = 0, 0
    for i in range(num_bins):
        ECE += (counts[i] / n) * abs(avg_acc[i] - avg_conf[i])
        OE += (counts[i] / n) * (avg_conf[i] * (max(avg_conf[i] - avg_acc[i], 0)))

    return ECE, OE, avg_acc, avg_conf, sum(acc_counts) / n, counts


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def save_checkpoint(path, net, acc, ece, oe, epoch, alpha):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'ece': ece,
        'oe': oe,
        'epoch': epoch,
        'alpha': alpha,
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, path+'/ckpt-{}.t7'.format(epoch))
    print('Saved model to {}'.format(path))


def load_checkpoint(filename=None, criterion=None):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    details = {}
    net = None
    keys = ['net', 'acc', 'ece', 'oe', 'epoch', 'rng_state', 'alpha']
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        for key in keys:
            if key == 'net':
                net = checkpoint['net']
            elif key == 'criterion' and criterion is not None:
                criterion.load_state_dict(checkpoint['optimizer'])
            else:
                try:
                    details[key] = checkpoint[key]
                except:
                    continue

        print("Loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
    print('Loaded Details {}'.format(details))
    return net, criterion, details


def plot_barplot(data, x, y, title, ylabel, xlabel, file, view=False):
    # sns.set(style="whitegrid")
    fig, ax = plt.subplots()
    sns.barplot(data=data, x=x, y=y, ax=ax, ci='sd', capsize=0.2)
    ax.set_title(title)
    ax.set_ylabel = ylabel
    ax.set_xlabel = xlabel
    if view:
        plt.show()
    plt.savefig(file)
    plt.close()


def plot_distplot(data, alphas, title, ylabel, xlabel, file, view=False):
    # sns.set(style="whitegrid")
    fig, ax = plt.subplots()
    for alpha in alphas:
        sns.distplot(data[alpha], ax=ax, hist=False, label='alpha={}'.format(alpha))
    ax.set_title(title)
    ax.set_ylabel = ylabel
    ax.set_xlabel = xlabel
    plt.xlim(0, 1.0)
    plt.legend()
    if view:
        plt.show()
    plt.savefig(file)
    plt.close()


def plot_scatterplot(x, y, title, ylabel, xlabel, file, weights, view=False):
    # sns.set(style="whitegrid")
    fig, ax = plt.subplots()
    sns.scatterplot(x, y, ax=ax, size=weights, sizes=(10, 200), alpha=0.6)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if view:
        plt.show()
    plt.savefig(file)
    plt.close()


def plot_joint(x, y, xlabel, ylabel, title, file, view=False):
    fig = plt.figure()
    g = sns.jointplot(x, y, kind='kde', color='b', xlim=(None, 1), ylim=(None, 1), shaded=False)
    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)
    g.fig.suptitle(title)
    if view:
        plt.show()
    plt.savefig(file)
    plt.close()
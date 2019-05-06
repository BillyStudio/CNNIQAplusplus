import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from IQADataset import IQADataset
from scipy import stats


try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class CNNIQAplusnet(nn.Module):
    def __init__(self, n_classes, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNNIQAplusnet, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kers, ker_size)
        self.fc1   = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2   = nn.Linear(n1_nodes, n2_nodes)
        self.fc3_q = nn.Linear(n2_nodes, 1)
        self.fc3_d = nn.Linear(n2_nodes, n_classes)

    def forward(self, x, training=True):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #
        h = self.conv1(x)

        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h = torch.cat((h1, h2), 1)  #
        h = h.squeeze(3).squeeze(2)

        h = F.relu(self.fc1(h))
        if training is True:
            h = F.dropout(h)
        h = F.relu(self.fc2(h))

        q = self.fc3_q(h)
        d = self.fc3_d(h)

        return q, d


class CNNIQAplusplusnet(nn.Module):
    def __init__(self, n_distortions, ker_size=3, n1_kers=8, pool_size=2, n2_kers=32, n1_nodes=128, n2_nodes=512):
        super(CNNIQAplusplusnet, self).__init__()
        self.conv1 = nn.Conv2d(1, n1_kers, ker_size)
        self.pool1 = nn.MaxPool2d(pool_size)
        self.conv2 = nn.Conv2d(n1_kers, n2_kers, ker_size)
        self.fc1   = nn.Linear(2 * n2_kers, n1_nodes)
        self.fc2   = nn.Linear(n1_nodes, n2_nodes)
        self.fc3_q = nn.Linear(n2_nodes, 1)
        self.fc3_d = nn.Linear(n2_nodes, n_distortions)

    def forward(self, x, training=True):
        #print('original input size:', x.shape)
        x  = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #
        h  = self.conv1(x)
        h  = self.pool1(h)

        h  = self.conv2(h)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h  = torch.cat((h1, h2), 1)  #
        h  = h.squeeze(3).squeeze(2)

        h  = F.relu(self.fc1(h))
        if training is True:
            h  = F.dropout(h)
        h  = F.relu(self.fc2(h))

        q  = self.fc3_q(h)
        d  = self.fc3_d(h)

        return q, d


def get_data_loaders(config, train_batch_size, exp_id=0):
    train_dataset = IQADataset(config, exp_id, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              num_workers=4)

    val_dataset = IQADataset(config, exp_id, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset)

    if config['test_ratio']:
        test_dataset = IQADataset(config, exp_id, 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset)

        return train_loader, val_loader, test_loader

    return train_loader, val_loader

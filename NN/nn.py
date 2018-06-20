import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import matplotlib.pyplot as plt
import numpy as np

LR = 0.0005
BATCH_SIZE = 10


def train_val_split(train_set):
    '''
    splits train set into 80:20 train and validation
    :param train_set:
    :return:
    '''
    train_len = len(train_set)
    indices = list(range(train_len))
    split = int((train_len * 0.2))
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)

    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=BATCH_SIZE, sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=BATCH_SIZE, sampler=validation_sampler)

    return train_loader, validation_loader


def weight_init(m):
    '''
    initial weight with xaviar.
    :param m:
    :return:
    '''
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data)


def train(model, optimizer, train_loader):
    '''
    trains a given model using optimizer update rule
    :param model:
    :param optimizer:
    :param train_loader:
    :return:
    '''
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model.forward(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def model_check(model, loader, test_or_val):
    '''
    test given model on given data set and logs results
    :param model:
    :param loader:
    :param test_or_val:
    :return:
    '''
    model.eval()
    test_loss = 0
    correct = 0
    length = len(loader.dataset)
    if test_or_val == "Validation" or test_or_val == "Train":
        length = len(loader.sampler.indices)

    for data, target in loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= length
    print('{:s} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_or_val, test_loss, correct, length, 100. * correct / length))
    return test_loss


class Neural_Net(nn.Module):
    '''
    2 hidden layer NN with Batch Normalization before activation function
    '''

    def __init__(self, image_size):
        super(Neural_Net, self).__init__()
        self.image_size = image_size
        self.h0 = nn.Linear(image_size, 50)
        self.h0_bn = nn.BatchNorm1d(50)
        self.h1 = nn.Linear(50, 20)
        self.h1_bn = nn.BatchNorm1d(20)
        self.h2 = nn.Linear(20, 2)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.h0_bn(self.h0(x)))
        x = F.relu(self.h1_bn(self.h1(x)))
        x = self.h2(x)
        return F.log_softmax(x)


def plot_avg(avg_train, avg_valid, epoch):
    '''
    plot average  loss train and validation.
    :param avg_train:
    :param avg_valid:
    :param type:
    :param epoch:
    :return:
    '''
    xs = np.arange(0, epoch, 2)
    plt.plot(avg_train, color="red", label="Average loss train")
    plt.plot(avg_valid, color="blue", label="Average loss validation")

    # Create a legend for the first line.
    first_legend = plt.legend(loc=1)

    # Add the legend manually to the current Axes.
    ax = plt.gca().add_artist(first_legend)
    plt.xlabel('Epoch Number')
    plt.ylabel('Average loss')
    plt.show()


def predict(model, data_loader):
    '''
    preduct output from given model and data loader.
    :param model:
    :param data_loader:
    :return:
    '''
    model.eval()
    with open('test.pred', 'w+') as f:
        for data, real_tag in data_loader:
            out = model(data)
            out = out.data.numpy()
            for sample in out:
                f.write(str(sample.argmax()) + '\n')


if __name__ == "__main__":

    model = Neural_Net(32 * 32 * 3)
    model.apply(weight_init)

    # transforms made on data
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    stenog_dataset = torchvision.datasets.ImageFolder(
        root='/home/daniel/Documents/stang_proj/stenography/ready train', transform=transforms)

    train_loader, val_loader = train_val_split(stenog_dataset)

    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_loss = []
    val_loss = []

    for epoch in range(1, 30 + 1):
        print('Epoch: ', epoch)
        train(model=model, optimizer=optimizer, train_loader=train_loader)
        train_loss.append(model_check(model=model, loader=train_loader, test_or_val="Train"))
        val_loss.append(model_check(model=model, loader=val_loader, test_or_val="Validation"))

    # Plot graphs
    plot_avg(train_loss, val_loss, epoch)

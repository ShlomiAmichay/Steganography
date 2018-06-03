import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import matplotlib.pyplot as plt
import numpy as np

LR = 0.01
BATCH_SIZE = 32


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
    print('\n{:s} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_or_val, test_loss, correct, length, 100. * correct / length))
    return test_loss


class FirstNet(nn.Module):
    '''
    2 hidden layer NN
    '''

    def __init__(self, image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.h0 = nn.Linear(image_size, 100)
        self.h1 = nn.Linear(100, 50)
        self.h2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.h0(x))
        x = F.relu(self.h1(x))
        x = self.h2(x)
        return F.log_softmax(x)


class DropoutNet(nn.Module):
    '''
    2 hidden layer NN with Dropout after activation function
    '''

    def __init__(self, image_size):
        super(DropoutNet, self).__init__()
        self.image_size = image_size
        self.h0 = nn.Linear(image_size, 100)
        self.h0_drop = nn.Dropout(p=0.25)
        self.h1 = nn.Linear(100, 50)
        self.h1_drop = nn.Dropout(p=0.25)
        self.h2 = nn.Linear(50, 10)
        self.h2_drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.h0_drop(F.relu((self.h0(x))))
        x = self.h1_drop(F.relu((self.h1(x))))
        x = self.h2_drop((self.h2(x)))
        return F.log_softmax(x)


class Batch_Norm_Net(nn.Module):
    '''
    2 hidden layer NN with Batch Normalization before activation function
    '''

    def __init__(self, image_size):
        super(Batch_Norm_Net, self).__init__()
        self.image_size = image_size
        self.h0 = nn.Linear(image_size, 100)
        self.h0_bn = nn.BatchNorm1d(100)
        self.h1 = nn.Linear(100, 50)
        self.h1_bn = nn.BatchNorm1d(50)
        self.h2 = nn.Linear(50, 10)


    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.h0_bn(self.h0(x)))
        x = F.relu(self.h1_bn(self.h1(x)))
        x = self.h2(x)
        return F.log_softmax(x)


def predict(model, data_loader):
    model.eval()
    with open('test.pred', 'w+') as f:
        for data, real_tag in data_loader:
            out = model(data)
            out = out.data.numpy()
            out = out.argmax()
            f.write(str(out) + '\n')


# model = FirstNet(28 * 28)

model = Batch_Norm_Net(32 * 32 * 3)

# model = DropoutNet(28 * 28)

# transforms made on data
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.1307,), (0.3081,))])
# FashionMNIST train and test data sets
# mnist_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transforms)
# mnist_test = torchvision.datasets.FashionMNIST('./data', train=False, transform=transforms)

stenog_dataset = torchvision.datasets.ImageFolder(
    root='/Users/shlomiamichay/Desktop/Project/Task4/Stenogarphy/stenography/ready train', transform=transforms)
# train_loader = utils.data.DataLoader(stenog_dataset,
#                                            batch_size=BATCH_SIZE, shuffle=True,
#                                            num_workers=4)
train_loader, test_loader = train_val_split(stenog_dataset)

# load data sets and divide to train and validation
# train_loader, val_loader = train_val_split(mnist_train)
# test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=True)

# set optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

train_loss = []
val_loss = []
test_loss = []

for epoch in range(1, 10 + 1):
    train(model=model, optimizer=optimizer, train_loader=train_loader)
    train_loss.append(model_check(model=model, loader=train_loader, test_or_val="Train"))
    # val_loss.append(test(model=model, loader=val_loader, test_or_val="Validation"))
    test_loss.append(model_check(model=model, loader=test_loader, test_or_val="Test"))

# make prediction on test with trained model
predict(model=model, data_loader=test_loader)

# Plot graphs
p = plt.figure()
s = plt.plot(range(10), val_loss, color="red", label="Validation Avg Loss")
t = plt.plot(range(10), test_loss, color="blue", label="Train Avg Loss")

# Create a legend for the lines
first_legend = p.legend(loc=1)

# Add the legend manually to the current Axes
# p.gca().add_artist(first_legend)

# Save figure
# p.savefig('plot.png')

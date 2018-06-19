import torch
from scipy import ndimage
from torch import nn, utils
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

BATCH_SIZE = 10
num_epochs = 1
device = 'cpu'


class ConvNet(nn.Module):
    '''
    3 hidden layer CNN with Batch Normalization before activation function and max pool after.
    '''

    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(3, 8, kernel_size=5),
            nn.BatchNorm2d(8),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=5, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=5, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(32, 100),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.Linear(50, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = F.log_softmax(out)
        return out


def high_pass_filter(data):
    '''
    high pass filter on given imege.
    :param data:
    :return:
    '''
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    highpass_3x3_r = ndimage.convolve(data[0, :, :], kernel)
    highpass_3x3_g = ndimage.convolve(data[1, :, :], kernel)
    highpass_3x3_b = ndimage.convolve(data[2, :, :], kernel)

    highpass_3x3 = np.array([highpass_3x3_r, highpass_3x3_g, highpass_3x3_b])

    return torch.from_numpy(highpass_3x3)


def weight_init(m):
    '''
    initial weight with xaviar.
    :param m:
    :return:
    '''
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data)


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


def test(model, loader, criterion):
    '''
    test given model on given data set and logs results
    :param model:
    :param loader:
    :param criterion:
    :return:
    '''
    model.eval()
    test_loss = 0
    length = len(loader.sampler.indices)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            test_loss += criterion(outputs, labels)  # sum up batch loss
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} % Avg loss: {}'.format(100 * correct / total,
                                                                                              test_loss / length))


def train(epoch, model, optimizer, train_loader, criterion):
    '''
       trains a given model using optimizer update rule
       :param epoch
       :param model:
       :param optimizer:
       :param train_loader:
       :param criterion
       :return:
       '''
    i = 0
    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels, size_average=False)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        if (i + 1) % 100 == 0:
            print('[Epoch {}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, i + 1, len(train_loader.sampler.indices), loss.item()))
        i += 1
        optimizer.step()


data_transform = transforms.Compose([transforms.ToTensor()])
stenog_dataset = datasets.ImageFolder(
    root='./ready train', transform=data_transform)

train_loader, test_loader = train_val_split(stenog_dataset)

model = ConvNet(2).cpu()
model.apply(weight_init)
# Loss and optimizer
criterion = F.nll_loss
optimizer = torch.optim.Adadelta(model.parameters())

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    i = 0
    train(epoch=epoch, model=model, optimizer=optimizer, train_loader=train_loader, criterion=criterion)

    test(model, train_loader, criterion)
    test(model, test_loader, criterion)

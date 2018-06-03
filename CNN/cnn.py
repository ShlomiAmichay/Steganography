import torch
from torch import nn, utils
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

BATCH_SIZE = 4
num_epochs = 5
device = 'cpu'


class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2),
            nn.BatchNorm2d(8),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=1, stride=1))
        # self.fc = nn.Linear(64, num_classes)
        # self.fc = nn.Linear(64, num_classes)
        self.fc =nn.Sequential(
        nn.Linear(16, 100),
        nn.BatchNorm1d(100),
        nn.Linear(100, 50),
        nn.BatchNorm1d(50),
        nn.Linear(50, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = F.log_softmax(out)
        return out


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


def test(model, optimizer, loader, criterion):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
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
    i = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        # o, ind = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        if (i + 1) % 100 == 0:
            print('[Epoch {}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, i + 1, len(train_loader), loss.item()))
        i += 1
        optimizer.step()


data_transform = transforms.Compose([transforms.ToTensor()])
stenog_dataset = datasets.ImageFolder(
    root='/Users/shlomiamichay/Desktop/Project/Task4/Stenogarphy/stenography/ready train', transform=data_transform)
# train_loader = utils.data.DataLoader(stenog_dataset,
#                                            batch_size=BATCH_SIZE, shuffle=True,
#                                            num_workers=4)
train_loader, test_loader = train_val_split(stenog_dataset)
model = ConvNet(2).cpu()

# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = F.nll_loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    i = 0
    train(epoch=epoch, model=model, optimizer=optimizer, train_loader=train_loader, criterion=criterion)

    test(model, optimizer, test_loader, criterion)

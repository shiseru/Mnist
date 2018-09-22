import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


SEED = 1
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000

torch.manual_seed(SEED) #user seed as randam parameter to shuffle image

#dataset for learning
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data',
        train=True,
        download=True,
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ),
    batch_size = BATCH_SIZE,
    shuffle=True,
    **kwargs
)

#dataset for testing
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data',
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ),
    batch_size=TEST_BATCH_SIZE,
    shuffle=True,
    **kwargs)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net().to(device)

LEARNING_RATE = 0.01
MOMENTUM = 0.5

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)




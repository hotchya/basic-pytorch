import torch.nn as nn

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU()
        )
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(2)

        # self.fc1 = nn.Linear(256, 120)
        # self.relu3 = nn.ReLU()
        # self.fc2 = nn.Linear(120, 84)
        # self.relu4 = nn.ReLU()
        # self.fc3 = nn.Linear(84, 10)
        # self.relu5 = nn.ReLU()
        

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
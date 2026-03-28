import torch
import torch.nn as nn
import torch.nn.functional as F

class tinyimg_CNET(nn.Module):
    def __init__(self):
        super(tinyimg_CNET, self).__init__()

        #convolution layers
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.conv2 = nn.Conv2d(64, 128, 5, 1, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 384, 5, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 1, 1, 1)

        #sropout layers
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        #fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 200)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)

        x = F.adaptive_avg_pool2d(x, (3, 3))

        x = torch.flatten(x, 1)

        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        output = F.softmax(x, dim=1)

        return output
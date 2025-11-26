# network.py - Модель тодорхойлолт
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3)
        self.fc1 = nn.Linear(512 * 4 * 4, 128)
        
    def forward(self, x):
        # 4 давхаргат CNN + fully connected
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 512 * 4 * 4)
        x = self.fc1(x)
        return x

class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.cnn = CNN()
        self.fc = nn.Linear(128, 1)
        
    def forward(self, x1, x2):
        # Хоёр зургийг feature extraction
        out1 = self.cnn(x1)
        out2 = self.cnn(x2)
        # Absolute difference
        diff = torch.abs(out1 - out2)
        out = self.fc(diff)
        return torch.sigmoid(out)

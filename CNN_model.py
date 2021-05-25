# %%

import torch
#import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


chars = np.load('data/data.npy')
labels = np.load('data/labels.npy')
device = torch.device("cuda")
data = torch.Tensor(chars).unsqueeze(1)
labels = torch.Tensor(labels).to(dtype = torch.long)


n_classes = max(labels).item() + 1
batch_size = 32
num_workers = 0


#_, axes = plt.subplots(nrows=1, ncols=5, figsize=(35, 5))
#for ax, image, label in zip(axes, chars, labels):
#    ax.set_axis_off()
#    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#    ax.set_title(f'Char n° {label}')


class MathsDataset(Dataset):
    def __init__(self, X, Y):
        self.images = X
        self.labels = Y
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return (self.images[i], self.labels[i])


n_samples = len(data)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=False)

train_loader = DataLoader(MathsDataset(X_train, y_train), batch_size=batch_size, num_workers=num_workers)

test_loader = DataLoader(MathsDataset(X_test, y_test), batch_size=batch_size, num_workers=num_workers)


# %%
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128 * 8 * 8, 500)
        self.fc2 = nn.Linear(500, n_classes + 1)
        self.dropout = nn.Dropout(0.25)
        
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.sigmoid(x)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features
        y = (1.0/np.sqrt(n))
        m.weight.data.normal_(0, y)
        m.bias.data.fill_(0)

# %%

model = Classifier()
model.apply(weights_init_normal)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

losses = {'train':[], 'test':[]}

def train(n_epochs, save_path):
    test_loss_min = np.inf
    for epoch in range(1, n_epochs+1):
        train_loss = 0.
        test_loss = 0.
        model.train()
        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.data 
            
        model.eval()
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.data
            
            losses['train'].append(train_loss)            
            losses['test'].append(test_loss)            
        print('Epoch: {} \tTraining Loss: {:.6f} \Test Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            test_loss
            ))
        if test_loss <= test_loss_min:
            print('Test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            test_loss_min,
            test_loss))
            torch.save(model.state_dict(), save_path)
            test_loss_min = test_loss
    return model         

def load_model(path):
    model = Classifier()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def model_accuracy(model):
    predicted = torch.argmax(model(data.to(device)), dim = 1).cpu().numpy()
    print(f"Classification report for classifier {model}:\n"
      f"{metrics.classification_report(labels, predicted)}\n")
# %%

#train(200,'models/CNN_1.pth')
model = load_model("models/CNN_1.pth")
#model_accuracy(model)
# %%

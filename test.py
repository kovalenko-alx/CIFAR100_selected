import pandas
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def conv_nc_2(weights, gamma):
    tensor_sum_nc = 0.0

    for i in range(len(weights)):
        for j in range(len(weights)):
            tensor_sum_nc += torch.mean(torch.matmul(weights[j], weights[i])) ** 2

    loss = gamma * tensor_sum_nc
    return loss


def tensor_to_float(tensor):
    lst = []

    for i in range(len(tensor)):
        lst.append(float(tensor[i]))

    return lst


def limited_set(data, target_list):
    features_lim = []
    labels_lim = []

    for i in range(len(data.targets)):

        if training_dataset.targets[i] in target_list:
            features_lim.append(data.data[i])
            labels_lim.append(data.targets[i])

    return np.asarray(np.transpose(features_lim, (0, 3, 1, 2))), np.asarray(labels_lim)


def full_set(data):

    features = data.data
    labels = data.targets

    return np.asarray(np.transpose(features, (0, 3, 1, 2))), np.asarray(labels)


class MyDataset(Dataset):

    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.fc1 = nn.Linear(4 * 4 * 128, 300)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(300, 100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 128)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


transform_train = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((32, 32)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
training_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
validation_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=100, shuffle=False)

features_new, labels_new = limited_set(training_dataset, list(range(10)))
features_test, labels_test = limited_set(validation_dataset, list(range(10)))

train_dataset = MyDataset(features_new, labels_new, transform=transform_train)
train_dataloader = DataLoader(train_dataset, batch_size = 100, shuffle=True)

test_dataset = MyDataset(features_test, labels_test, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size = 100, shuffle=False)

model = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []
dots = []

for e in range(epochs):

    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0

    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        # print(my_loss_dot(model.fc1.weight, 0.01))
        dot = conv_nc_2(model.conv2.weight, 1) + conv_nc_2(model.conv2.weight, 1)
        loss = criterion(outputs, labels)# + dot

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    else:
        with torch.no_grad():
            for val_inputs, val_labels in test_dataloader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)# + dot

                _, val_preds = torch.max(val_outputs, 1)
                val_running_loss += val_loss.item()
                val_running_corrects += torch.sum(val_preds == val_labels.data)

        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = running_corrects.float() / len(train_dataloader)
        running_loss_history.append(epoch_loss)
        running_corrects_history.append(epoch_acc)
        dots.append(dot)

        val_epoch_loss = val_running_loss / len(test_dataloader) / 10
        val_epoch_acc = val_running_corrects.float() / len(test_dataloader)*10
        val_running_loss_history.append(val_epoch_loss)
        val_running_corrects_history.append(val_epoch_acc)
        print('epoch :', (e + 1))
        print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
        print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))
        print('dot: {:.4f} '.format(dot.item()))

list1 = tensor_to_float(running_corrects_history)
list2 = tensor_to_float(val_running_corrects_history)
list3 = tensor_to_float(dots)
list_range = list(range(1, len(list1) + 1))

df = pandas.DataFrame(data={"epoch": list_range, "train": list1, "test": list2, "dots": list3})
df.to_csv("./CIFAR100_LeNet2L_reference_50_epoch.csv", sep=',', index=False)

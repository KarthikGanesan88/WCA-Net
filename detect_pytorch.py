import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from tqdm.auto import tqdm

def accuracy(model, data_loader, device):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data = data.to(device)
            target = target.to(device)
            _, predictions = torch.max(model(data), 1)
            correct += (predictions == target).sum().item()
            total += target.size(0)
    return correct/total


class customDataset(Dataset):
    def __init__(self, dset, labels):
        self.images = dset
        self.targets = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]

def createDataLoaders(clean_data, FGSM_data, PGD_data):
    labels = torch.cat((torch.zeros(len(clean_data)),
                        torch.ones(len(FGSM_data) + len(PGD_data))))
    labels = labels.type(torch.long)
    dataset = customDataset(dset=torch.cat((clean_data, FGSM_data, PGD_data)), labels=labels)

    total = len(dataset)
    train_split = int(0.7 * total)
    test_split = total - train_split
    trainset, testset = random_split(dataset, [train_split, test_split],
                                     generator=torch.Generator().manual_seed(42))
    trainloader = DataLoader(trainset, batch_size=128, shuffle=False, num_workers=0)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
    return trainloader, testloader

class detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), stride=(1, 1))
        self.mp_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv_2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5), stride=(1, 1))
        self.mp_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc_1 = nn.Linear(in_features=300, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=2)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.mp_1(x)
        x = F.relu(self.conv_2(x))
        x = self.mp_2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

device = torch.device('cuda')

cifar10 = torch.load('./images/cifar10.pt')
FGSM_baseline = torch.load('./images/FGSM_baseline.pt')
PGD_baseline = torch.load('./images/PGD_baseline.pt')
FGSM_robust = torch.load('./images/FGSM_robust.pt')
PGD_robust = torch.load('./images/PGD_robust.pt')

bs_trainloader, bs_testloader = createDataLoaders(cifar10, FGSM_baseline, PGD_baseline)
# rb_trainloader, rb_testloader = createDataLoaders(cifar10, FGSM_robust, PGD_robust)

model = detector()
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
loss_func = nn.CrossEntropyLoss()

print('Starting training.')
for epoch in range(10):
    for data, target in tqdm(bs_trainloader, leave=False):
        data = data.to(device)
        target = target.to(device)
        model.train()
        logits = model(data)
        optimizer.zero_grad()
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
    train_acc = accuracy(model, bs_trainloader, device=device)
    test_acc = accuracy(model, bs_testloader, device=device)
    print(f'Epoch:{epoch}. Train accuracy: {100.*train_acc:.2f}%. Test accuracy: {100.*test_acc:.2f}%.')



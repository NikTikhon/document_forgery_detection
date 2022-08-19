import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
import argparse



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        nn.init.xavier_uniform_(self.conv1.weight)

        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        nn.init.xavier_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        nn.init.xavier_uniform_(self.conv3.weight)

        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        nn.init.xavier_uniform_(self.conv4.weight)

        self.conv5 = nn.Conv2d(64, 128, 3, 1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        nn.init.xavier_uniform_(self.conv5.weight)

        self.conv6 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        nn.init.xavier_uniform_(self.conv6.weight)
        self.bn7 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(8192, 128, bias=False)
        self.fc2 = nn.Linear(128, 2)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.dropout4 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.pool(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.dropout3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    loss_out = 0
    for batch, (X, y) in enumerate(dataloader):

        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_out += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fake detection")

    main_arg_parser.add_argument("--epochs", type=int, default=50,
                                  help="number of training epochs, default is 50")
    main_arg_parser.add_argument("--batch-size", type=int, default=16,
                                  help="batch size for training, default is 16")
    main_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    main_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    args = main_arg_parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    transform = transforms.Compose([transforms.ToTensor()])
    full_set = datasets.ImageFolder(args.dataset, transform=transform)
    train_size = int(0.9 * len(full_set))
    val_size = len(full_set) - train_size
    train_set, test_set = torch.utils.data.random_split(full_set, [train_size, val_size])
    testloader = DataLoader(test_set, shuffle=True, batch_size=args.batch_size)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size)
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    for t in range(args.epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, net, criterion, optimizer, device)
        test_loop(testloader, net, criterion, device)
    torch.save(net.state_dict(), args.save_model_dir)
if __name__ == '__main__':
    main()

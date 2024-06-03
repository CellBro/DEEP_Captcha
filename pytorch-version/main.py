'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import os
import argparse
import time
from models import *
from utils import progress_bar
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--b', default=16, type=int, help='Batch size')
parser.add_argument('--e', default=100, type=int, help='Batch size')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
BATCH_SIZE = args.b
learning_rate = args.lr
EPOCH=args.e
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

transform_test = transforms.Compose([
transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataset_train = datasets.ImageFolder('../torch_data/train', transform_train)
dataset_val = datasets.ImageFolder('../torch_data/test', transform_test)

trainloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)

f = open(f"./logs/Resnet18_lr_{learning_rate}.log", 'w')
f.write(f"Current batch_size is {BATCH_SIZE}\n")
f.write(f"Current Learning Rate is {learning_rate}\n\n")
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/Resnet18_lr_{learning_rate}_ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    f.write('Training : Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
            % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    f.write('Testing : Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        f.write(f"Current Epoch has the best ACC : {acc}\n")
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/Resnet18_lr_{learning_rate}_ckpt.pth')
        best_acc = acc
    f.write(f"Best ACC so far: {best_acc}\n")


train_time = 0.0
for epoch in range(start_epoch, start_epoch + EPOCH):
    f.write(f"Epoch {epoch}\n")
    train_start = time.perf_counter()
    train(epoch)
    test(epoch)
    scheduler.step()
    train_end = time.perf_counter()
    epoch_time = train_end - train_start
    f.write(f"当前Epoch用时: {epoch_time :.6f} s\n")
    print(f"当前Epoch用时: {epoch_time :.6f} s")
    train_time = train_time + epoch_time
    f.write(f"目前为止已经过: {train_time :.6f} s\n\n")
    print(f"目前为止已经过: {train_time :.6f} s")
    f.flush()
f.write("Total training time:{}".format(train_time))
print("Total training time:{}".format(train_time))

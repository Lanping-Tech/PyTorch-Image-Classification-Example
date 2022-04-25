import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.backends.cudnn as cudnn
import argparse

from sklearn.metrics import classification_report 
from tqdm import tqdm
import models
from utils import performance_display

import warnings
warnings.filterwarnings("ignore")

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='PyTorch Image Classification Example'
    )

    parser.add_argument('--model-name', type=str, default='resnet50', help='model name')

    # dataset
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N', help='input batch size for testing (default: 100)')

    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help='divice')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')


    parser.add_argument('--output_path', default="output", type=str, help='output path')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    
    return parser.parse_args()

def test(model, data_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    item_num = 0
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            y_true_all.append(labels.cpu().numpy())
            y_pred_all.append(predicted.cpu().numpy())
            item_num += labels.size(0)
    test_loss /= len(data_loader)
    correct /= item_num
    print('Test Loss: %.4f' % (test_loss))
    print('Test Accuracy: %.4f' % (correct))
    y_pred_all = np.concatenate(y_pred_all)
    y_true_all = np.concatenate(y_true_all)
    print(classification_report(y_true_all, y_pred_all))
    return test_loss, correct

def train(model, data_loader, optimizer, criterion, device, args):
    model.train()
    train_loss = 0
    correct = 0
    item_num = 0
    pbar = tqdm(data_loader)
    for data in pbar:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        item_num += labels.size(0)
    train_loss /= len(data_loader)
    correct /= item_num
    print('Train Loss: %.4f' % (train_loss))
    print('Train Accuracy: %.4f' % (correct))
    return train_loss, correct

def main(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # 定义数据集
    train_data = torchvision.datasets.ImageFolder(root='Fruit_data/Train', transform=transform)
    test_data = torchvision.datasets.ImageFolder(root='Fruit_data/Test', transform=transform)
    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_load = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)

    # 定义模型
    model = getattr(models, 'get_'+args.model_name)(num_classes=len(train_data.classes)).to(args.device)
    print(model)

    optimizor = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0
    bad_num = 0
    # 开始训练
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_load, optimizor, criterion, args.device, args)
        test_loss, test_acc = test(model, test_load, criterion, args.device)
        print('Epoch: %d, Train Loss: %.4f, Train Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f' % (epoch, train_loss, train_acc, test_loss, test_acc))
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        if args.save_model:
            torch.save(model.state_dict(), '%s/%s_model_%d.pth' % (args.output_path,args.model_name, epoch))
        if best_acc < train_acc:
            bad_num = 0
            print('Saving best model...')
            torch.save(model.state_dict(), '{}_best_model.pth'.format(args.model_name))
            best_acc = train_acc
        else:
            bad_num += 1

        if bad_num > 10:
            break

    # 绘制训练集和测试集的损失和准确率
    loss_plot = {}
    loss_plot['train_loss'] = train_losses
    loss_plot['test_loss'] = test_losses

    acc_plot = {}
    acc_plot['train_acc'] = train_accs
    acc_plot['test_acc'] = test_accs
    performance_display(acc_plot, "ACC", args.output_path)
    performance_display(loss_plot, "LOSS", args.output_path)

if __name__ == "__main__":
    seed_torch()
    args = parse_arguments()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    main(args)

    # nohup python -u main.py > train.log 2>&1 &

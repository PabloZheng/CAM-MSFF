import numpy as np
import argparse
from time import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from network.net import VGG, make_layers
from network.utils import CatGAP, accuracy


def compute_loss(outlist, labels):
    criterion = nn.CrossEntropyLoss()
    loss_dic = {}
    for i in range(5):
        loss_dic[str(i)] = criterion(outlist[i], labels)
    fusion_loss = criterion(outlist[5], labels)
    loss = loss_dic['0'] + loss_dic['1'] + loss_dic['2'] + loss_dic['3'] + loss_dic['4'] + fusion_loss
    return loss, loss_dic['4'], fusion_loss


def create_model(device, num_cls):
    model = VGG(make_layers(), num_cls)
    model.to(device)
    model.classifier = nn.Sequential(CatGAP())
    print(model)
    return model


def train(model, opt):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(opt.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(opt.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset_train = datasets.ImageFolder(root=opt.train_path, transform=train_transforms)
    dataset_valid = datasets.ImageFolder(root=opt.val_path, transform=valid_transforms)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    for epoch in range(100):
        if epoch % 5 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9
    minLoss,  maxValacc= 99999, -99999
    cls_wtbox = {}
    for i in range(7):
        cls_wtbox[str(i)] = []
    for epoch in range(opt.epochs):
        print('EPOCH: ', epoch + 1, '/%s' % opt.epochs)
        model.train()
        # initialize work
        train_acc, val_acc = [], []
        running_loss, running_loss4, running_fusionloss = 0.0, 0.0, 0.0
        outlilst_value, stepTime = [], []
        step_acc, step_loss, step_loss4, count = 0, 0, 0, 0
        step_fusionloss, val_running_loss, val_running_loss4, val_running_fusionloss = 0.0, 0.0, 0.0, 0.0
        # initialize step attention weights save
        step_cls_wtbox = {}
        for i in range(7):
            step_cls_wtbox[str(i)] = []
        for images, labels in train_loader:
            step_start = time()
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            labels_index = labels.detach().cpu().numpy().tolist()
            outlist, scale_w = model(images)
            scale_w = scale_w.detach().squeeze().cpu().numpy()
            # step attention weights_save
            for i in range(len(labels_index)):
                step_cls_wtbox[str(labels_index[i])].append(scale_w[i].tolist())
            loss, loss4, fusion_loss = compute_loss(outlist, labels)
            outlilst_value.append(outlist[4])
            step_point_acc = accuracy(outlist[4], labels)
            step_acc += step_point_acc
            step_loss += loss.detach().cpu().numpy()
            train_acc.append(accuracy(outlist[4], labels))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss4 += loss4.item()
            running_fusionloss += fusion_loss.item()
            count += 1
            step_end = time()
            stepTime.append(step_end-step_start)
        # compute average
        for i in range(7):
            att_w = np.array(step_cls_wtbox[str(i)])
            att_w = att_w.sum(0)/len(step_cls_wtbox[str(i)])
            cls_wtbox[str(i)].append(att_w)
        print('Training sum loss:.......', running_loss / count)
        print('Training end loss:.......', running_loss4 / count)
        print('Training fusion loss:.......', running_fusionloss / count)
        with torch.no_grad():
            model.eval()
            count = 0
            for images, labels in valid_loader:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
                outlist, scale_w = model(images)
                loss, loss4, fusion_loss = compute_loss(outlist, labels)
                val_acc.append(accuracy(outlist[4], labels))
                val_running_loss += loss.item()
                val_running_loss4 += loss4.item()
                val_running_fusionloss += fusion_loss.item()
                count += 1
            mean_val_loss = val_running_loss / count
            mean_val_loss4 = val_running_loss4/count
            mean_val_fusionloss = val_running_fusionloss/count
            val_acc_ = np.mean(val_acc)
            print('Validation sum loss:.....', mean_val_loss)
            print('Validation end loss:.....', mean_val_loss4)
            print('Validation fusion loss:.....', mean_val_fusionloss)
            print('Training accuracy:...', np.mean(train_acc))
            print('Validation accuracy..', val_acc_)

            if mean_val_loss < minLoss:
                torch.save(model.state_dict(), opt.weight_path + '/best_loss.pth')
                print(f'NEW BEST Val Loss: {mean_val_loss} ........old best:{minLoss}')
                minLoss = mean_val_loss
                print('')
            if val_acc_ > maxValacc:
                torch.save(model.state_dict(), opt.weight_path + '/best_acc.pth')
                print(f'NEW BEST Val Acc: {val_acc_} ........old best:{maxValacc}')
                maxValacc = val_acc_
            print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-path', type=str, default='./', help='path for saving weights')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size used for training')
    parser.add_argument('--input-size', type=int, default=224, help='the size of input image')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs used for training')
    parser.add_argument('--train-path', type=str, default="./dataset/train/", help='path of training dataset')
    parser.add_argument('--val-path', type=str, default="./dataset/val/", help='path of validation dataset')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial loss value used for training')
    parser.add_argument('--gpu-ids', type=str, default='0', help='the index of device used for acceleration')
    opt = parser.parse_args()

    device = torch.device(f"cuda:{opt.gpu_ids}" if torch.cuda.is_available() else "cpu")
    train(create_model(device, num_cls=7), opt)
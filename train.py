import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.BCNN import BCNN
from data_utils.data_loader import FGVC_Dataset

import logging
logging.basicConfig(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/', type=str)
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--valid', action='store_true', help='validation')
    parser.add_argument('--early_stop', default=3, type=int, help='early stop')
    parser.add_argument('--save_dir', default='./weights/', type=str, help='save dir')

    return parser.parse_args()


def train(model, criterion, device, train_loader, optimizer):
    train_loss, train_acc = 0, 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pred = torch.max(outputs, dim=1)[1]
        train_acc += torch.mean((pred == labels).float()).item()
    
    train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_loader)

    return train_loss, train_acc


def validate(model, criterion, device, val_loader):
    valid_loss, valid_acc = 0, 0
    
    model.eval()
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        valid_loss += loss.item()

        pred = torch.max(outputs, 1)[1]
        # correct = pred.eq(labels).sum().item()
        acc = torch.mean((pred == labels).float())
        valid_acc += acc.item()
    
    valid_loss = valid_loss / len(val_loader)
    valid_acc = valid_acc / len(val_loader)
    
    return valid_loss, valid_acc


if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('Train device: {}'.format(device))
    logging.info('Args: {}'.format(args))

    # data preparation
    train_loader = FGVC_Dataset(args.data, is_train=True)
    val_loader = FGVC_Dataset(args.data, is_train=False)

    train_set = DataLoader(train_loader, batch_size=args.bs, shuffle=True)
    val_set = DataLoader(val_loader, batch_size=args.bs, shuffle=True)
    logging.info('Data loaded, train set: {}, val set: {}'.format(len(train_set), len(val_set)))

    # model preparation
    model = BCNN(fine_tune=False).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid, count = 0, 0
    logging.info('Start training')
    for epoch in range(args.epochs):
        print('Epoch: {}/{}'.format(epoch + 1, args.epochs))
        t_loss, t_acc = train(model, criterion, device, train_set, optimizer)
        print('Train Loss: {:.4f} Train Acc: {:.4f}'.format(t_loss, t_acc))

        if args.valid:
            v_loss, v_acc = validate(model, criterion, device, val_set)
            print('Validation Loss: {:.4f} Validation Acc: {:.4f}'.format(v_loss, v_acc))

        if v_acc > best_valid:
            best_valid = v_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
            
        else:
            count += 1
            if count >= args.early_stop:
                print('Early stopping at epoch {}'.format(epoch + 1))
                break

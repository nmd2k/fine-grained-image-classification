import os
import argparse
from tqdm import tqdm

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

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
    parser.add_argument('--early_stop', default=3, type=int, help='early stop')
    parser.add_argument('--save_dir', default='./weights/', type=str, help='save dir')
    parser.add_argument('--weight', default=None, type=str, help='weight')
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')

    return parser.parse_args()


def train(model, criterion, device, train_loader, optimizer):
    train_loss = 0
    train_acc = []
    for images, labels in (pbar := tqdm(train_loader, leave=False)):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pred = torch.max(outputs, dim=1)[1]
        correct = (pred == labels).sum().item()/len(labels)
        train_acc.append(correct)

        pbar.set_postfix(loss=f'{loss.item():.3f}', acc=f'{correct:.3f}')
    
    train_loss = train_loss / len(train_loader)
    train_acc = sum(train_acc) / len(train_acc)

    return train_loss, train_acc


def validate(model, criterion, device, val_loader):
    valid_loss = 0
    valid_acc = []
    
    model.eval()
    for images, labels in (pbar := tqdm(val_loader, leave=False)):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        valid_loss += loss.item()

        pred = torch.max(outputs, 1)[1]
        correct = (pred == labels).sum().item()/len(labels)
        valid_acc.append(correct)

        pbar.set_postfix(loss=f'{loss.item():.3f}', acc=f'{correct:.3f}')
    
    valid_loss = valid_loss / len(val_loader)
    valid_acc = sum(valid_acc) / len(valid_acc)
    
    return valid_loss, valid_acc


if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logging.info('Train device: {}'.format(device))
    logging.info('Args: {}'.format(args))
    wb = wandb.init(project='fine-grained-classification',)
    wb.config.update(args)

    torch.manual_seed(42)
    # data preparation
    train_set = FGVC_Dataset(args.data, is_train=True)
    train_len = int(len(train_set) * 0.8)
    val_len = len(train_set) - train_len
    train_set, val_set = random_split(train_set, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=True)
    logging.info('Data loaded, train set: {}, val set: {}'.format(train_len, val_len))

    # model preparation
    logging.info('Model initializating')
    model = BCNN().to(device)
    if args.weight is not None:
        model.load_state_dict(torch.load(args.weight))
        logging.info('Model loaded from {}'.format(args.weight))

    wb.watch(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('Model BCNN has: {} learnable params'.format(total_params))

    wb.config['total_learnable_params'] = total_params

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay, momentum=0.9)

    best_valid, count = 0, 0
    logging.info('Start training')
    for epoch in range(args.epochs):
        print('Epoch: {}/{}'.format(epoch + 1, args.epochs))
        t_loss, t_acc = train(model, criterion, device, train_loader, optimizer)
        print('Train Loss: {:.4f} Train Acc: {:.4f}'.format(t_loss, t_acc))

        v_loss, v_acc = validate(model, criterion, device, val_loader)
        print('Validation Loss: {:.4f} Validation Acc: {:.4f}'.format(v_loss, v_acc))

        if v_acc > best_valid:
            count = 0
            best_valid = v_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
            
        else:
            count += 1
            if count >= args.early_stop:
                print('Early stopping at epoch {}'.format(epoch + 1))
                break
        
        wb.log({'train/loss': t_loss, 
                'train/acc': t_acc, 
                'valid/loss': v_loss, 
                'valid/acc': v_acc},)
    
    wb.run.summary['best_valid_acc'] = best_valid
    logging.info('Training finished')

import os
import argparse
from tqdm import tqdm

import torch

from model.BCNN import BCNN
from data_utils.data_loader import FGVC_Dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/', help='path to dataset')
    parser.add_argument('--weight', type=str, default='./weights/best_model.pt', help='path to weight')

    return parser.parse_args()


def test(model, device, test_loader, classes):
    correct_pred = {classname:0 for classname in classes}
    total_pred = {classname:0 for classname in classes}

    with torch.no_grad():
        model.eval()
        for data, labels in tqdm(test_loader):
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            preds = torch.max(output, 1)[1]

            for label, pred in zip(labels, preds):
                if label == pred:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    avg = 0
    for classname, correct_count in correct_pred.items():
        acc = 100 * float(correct_count) / total_pred[classname]
        avg += acc
        print(f'Accuracy of {classname}: {acc:.1f}')

    avg /= len(correct_pred)
    print(f'===============================\nAverage accuracy: {avg}')

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data prepare
    test_set = FGVC_Dataset(args.data, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    with open(os.path.join(args.data, 'variants.txt'), 'r') as f:
        classes = f.read().splitlines()

    # model prepare
    model = BCNN().to(device)
    model.load_state_dict(torch.load(args.weight))

    # test
    test(model, device, test_loader, classes)

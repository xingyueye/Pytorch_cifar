import os
import sys
import torch
import model.darts.utils as utils
import logging
import argparse
import torch.nn as nn
import model.darts.genotypes as genotypes

from torch.autograd import Variable
from model.darts.model import NetworkCIFAR as Network
from loader import get_test_provider
import pandas as pd

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data/cifar', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='checkpoint/weights_9764.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto_augment')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS_V2', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()
    utils.load(model, args.model_path)

    test_loader, id_to_class = get_test_provider(args.batch_size)

    pred_labels = list()
    indices = list()
    model.eval()
    for data, fname in test_loader:
        data = data.cuda()
        with torch.no_grad():
            scores = model(data)
        labels = scores[0].max(1)[1].cpu().numpy()
        pred_labels.extend(labels)
        indices.extend(fname.numpy())
    df = pd.DataFrame({'id': indices, 'label': pred_labels})
    df['label'] = df['label'].apply(lambda x: id_to_class[x])
    df = df.sort_values(by='id', axis=0, ascending=True)
    df.to_csv('outputs/submission_train.csv', index=False)

if __name__ == '__main__':
    main()

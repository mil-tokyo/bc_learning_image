import os
import argparse


def parse():
    parser = argparse.ArgumentParser(description='BC learning for image classification')

    # General settings
    parser.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100'])
    parser.add_argument('--netType', required=True, choices=['convnet'])
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--nTrials', type=int, default=10)
    parser.add_argument('--save', default='None', help='Directory to save the results')
    parser.add_argument('--gpu', type=int, default=0)

    # Learning settings
    parser.add_argument('--BC', action='store_true', help='BC learning')
    parser.add_argument('--plus', action='store_true', help='Use BC+')
    parser.add_argument('--nEpochs', type=int, default=-1)
    parser.add_argument('--LR', type=float, default=-1, help='Initial learning rate')
    parser.add_argument('--schedule', type=float, nargs='*', default=-1, help='When to divide the LR')
    parser.add_argument('--warmup', type=int, default=-1, help='Number of epochs to warm up')
    parser.add_argument('--batchSize', type=int, default=-1)
    parser.add_argument('--weightDecay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    opt = parser.parse_args()
    if opt.plus and not opt.BC:
        raise Exception('Using only --plus option is invalid.')

    # Dataset details
    if opt.dataset == 'cifar10':
        opt.nClasses = 10
    else:  # cifar100
        opt.nClasses = 100

    # Default settings
    default_settings = dict()
    default_settings['cifar10'] = {
        'convnet': {'nEpochs': 250, 'LR': 0.1, 'schedule': [0.4, 0.6, 0.8], 'warmup': 0, 'batchSize': 128}
    }
    default_settings['cifar100'] = {
        'convnet': {'nEpochs': 250, 'LR': 0.1, 'schedule': [0.4, 0.6, 0.8], 'warmup': 0, 'batchSize': 128}
    }
    for key in ['nEpochs', 'LR', 'schedule', 'warmup', 'batchSize']:
        if eval('opt.{}'.format(key)) == -1:
            setattr(opt, key, default_settings[opt.dataset][opt.netType][key])

    if opt.save != 'None' and not os.path.isdir(opt.save):
        os.makedirs(opt.save)

    display_info(opt)

    return opt


def display_info(opt):
    if opt.BC:
        if opt.plus:
            learning = 'BC+'
        else:
            learning = 'BC'
    else:
        learning = 'standard'

    print('+------------------------------+')
    print('| CIFAR classification')
    print('+------------------------------+')
    print('| dataset  : {}'.format(opt.dataset))
    print('| netType  : {}'.format(opt.netType))
    print('| learning : {}'.format(learning))
    print('| nEpochs  : {}'.format(opt.nEpochs))
    print('| LRInit   : {}'.format(opt.LR))
    print('| schedule : {}'.format(opt.schedule))
    print('| warmup   : {}'.format(opt.warmup))
    print('| batchSize: {}'.format(opt.batchSize))
    print('+------------------------------+')

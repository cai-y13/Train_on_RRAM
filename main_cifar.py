import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import AverageMeter, Logger
from options import Options
from models.rram import get_rram_param_groups
from sparse_sgd import *
import models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

options = Options(description='PyTorch ImageNet Sparse Training')

options.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                     choices=model_names,
                     help='model architecture: ' +
                          ' | '.join(model_names) +
                          ' (default: resnet20)')
options.set_defaults(data='gen', epochs=164, batch_size=128, lr=0.1,
                     momentum=0.9, nesterov=False, weight_decay=1e-4,
                     lr_decay=0.1, lr_decay_step='80,121')
best_prec1 = 0


def main():
    global args, best_prec1, train_logger, test_logger, frequency_logger
    args = options.parse_args()
    config = args.get_config()
    os.makedirs(args.log_dir)
    os.makedirs(args.checkpoint_dir)
    train_logger = Logger(os.path.join(args.log_dir, 'train.log'))
    test_logger = Logger(os.path.join(args.log_dir, 'test.log'))
    frequency_logger = Logger(os.path.join(args.log_dir, 'frequency.log'))
    with open(os.path.join(args.log_dir, 'config.log'), 'w') as f:
        f.write(args.config_str)

    loss_results, top1_results, top5_results = torch.FloatTensor(args.epochs), torch.FloatTensor(args.epochs), \
                                               torch.FloatTensor(args.epochs)
    if args.visdom:
        from visdom import Visdom
        viz = Visdom()
        opts = [dict(title=args.experiment_id + ' Loss', ylabel='Loss', xlabel='Epoch'),
                dict(title=args.experiment_id + ' Top-1', ylabel='Top-1', xlabel='Epoch'),
                dict(title=args.experiment_id + ' Top-5', ylabel='Top-5', xlabel='Epoch')]
        viz_windows = [None, None, None]
        epochs = torch.arange(0, args.epochs)

    # create model
    if args.pretrained:
        print("=> using pre-trained model from {} '{}'".format(args.pretrained, args.arch))
        if 'fixed' in args.arch:
            model = models.__dict__[args.arch](fixed_bits={'input': args.input_bits,
                                                           'weight': args.weight_bits,
                                                           'output': args.output_bits})
        else:
            model = models.__dict__[args.arch]()
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> creating model '{}'".format(args.arch))
        if 'fixed' in args.arch:
            model = models.__dict__[args.arch](fixed_bits={'input': args.input_bits,
                                                           'weight': args.weight_bits,
                                                           'output': args.output_bits})
        else:
            model = models.__dict__[args.arch]()

    model = model.cuda() #torch.nn.DataParallel(model, device_ids=[i for i in range(args.nGPU)]).cuda()
    print(model)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.method == 'dft':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.method == 'max':
        optimizer = SparseMaxSGD(get_rram_param_groups(model.named_parameters()),
                                 args.lr, momentum=args.momentum,
                                 weight_decay=args.weight_decay,
                                 balanced_row_num={'conv': args.conv_balanced_row_num,
                                                   'fc': args.fc_balanced_row_num},
                                 balanced_freq={'conv':args.conv_balanced_freq,
                                                'fc':args.fc_balanced_freq},
                                 rram_size=args.rram_size,
                                 small_rram_size=args.small_rram_size,
                                 is_small_by_pos=args.is_small_by_pos)
    elif args.method == 'sum':
        optimizer = SparseSumSGD(get_rram_param_groups(model.named_parameters()),
                                 args.lr, momentum=args.momentum,
                                 weight_decay=args.weight_decay,
                                 balanced_row_num={'conv': args.conv_balanced_row_num,
                                                   'fc': args.fc_balanced_row_num},
                                 balanced_freq={'conv':args.conv_balanced_freq,
                                                'fc':args.fc_balanced_freq},
                                 rram_size=args.rram_size,
                                 small_rram_size=args.small_rram_size,
                                 is_small_by_pos=args.is_small_by_pos)
    elif args.method == 'lru':
        optimizer = SparseLRUSGD(get_rram_param_groups(model.named_parameters()),
                                 args.lr, momentum=args.momentum,
                                 weight_decay=args.weight_decay,
                                 rram_size=args.rram_size)
    elif args.method == 'fmx':
        optimizer = SparseMaxAccSGD(get_rram_param_groups(model.named_parameters()),
                                 args.lr, momentum=args.momentum,
                                 weight_decay=args.weight_decay,
                                 balanced_row_num={'conv': args.conv_balanced_row_num,
                                                   'fc': args.fc_balanced_row_num},
                                 balanced_freq={'conv':args.conv_balanced_freq,
                                                'fc':args.fc_balanced_freq},
                                 rram_size=args.rram_size,
                                 small_rram_size=args.small_rram_size,
                                 is_small_by_pos=args.is_small_by_pos,
                                 fixed_bits={'input': args.input_bits, 'weight':args.weight_bits})
    else:
        print('Unknown optimization method!')
        raise NameError

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            loss_results, top1_results, top5_results = checkpoint['loss_results'], checkpoint['top1_results'], \
                                                       checkpoint['top5_results']
            # Add previous scores to visdom graph
            if args.visdom and loss_results is not None:
                x_axis = epochs[0:args.start_epoch]
                y_axis = [loss_results[0:args.start_epoch], top1_results[0:args.start_epoch],
                          top5_results[0:args.start_epoch]]
                for x in range(len(viz_windows)):
                    viz_windows[x] = viz.line(
                        X=x_axis,
                        Y=y_axis[x],
                        opts=opts[x],
                    )
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])
    # normalize = transforms.Normalize(mean=[125.3, 123.0, 113.9],
    #                                  std=[63.0, 62.1, 66.7])

    train_dataset = datasets.CIFAR10(
        root=args.data, train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=args.data, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        train_logger.close()
        test_logger.close()
        return

    for epoch in range(args.start_epoch, args.epochs):

        print('='*89)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        loss = train(train_loader, model, criterion, optimizer, epoch)

        # log frequency
        if args.is_log_frequency:
            print('='*89)
            max_frequency_str = ','.join(map(lambda x: str(max(x)) if isinstance(x, list)
                                else str(torch.max(x)), optimizer.frequency_list))
            print('Max Number of Writing Operations:\n{}'.format(max_frequency_str))
            frequency_logger.write('{}\n'.format(max_frequency_str))
            if (epoch % args.log_freq == 0 or epoch == args.epochs-1):
                log_frequency(optimizer, epoch)

        # evaluate on validation set
        print('='*89)
        prec1, prec5 = validate(val_loader, model, criterion)

        loss_results[epoch] = loss
        top1_results[epoch] = prec1
        top5_results[epoch] = prec5

        if args.visdom:
            x_axis = epochs[0:epoch + 1]
            y_axis = [loss_results[0:epoch + 1], top1_results[0:epoch + 1],
                      top5_results[0:epoch + 1]]
            for x in range(len(viz_windows)):
                if viz_windows[x] is None:
                    viz_windows[x] = viz.line(
                        X=x_axis,
                        Y=y_axis[x],
                        opts=opts[x],
                    )
                else:
                    viz.line(
                        X=x_axis,
                        Y=y_axis[x],
                        win=viz_windows[x],
                        update='replace',
                    )

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'loss_results': loss_results,
            'top1_results': top1_results,
            'top5_results': top5_results,
        }, is_best)

    train_logger.close()
    test_logger.close()
    print('='*89)
    print('best_prec1: {}'.format(best_prec1))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
    print('Epoch: [{0}]\tLoss {loss.avg:.4f}\tPrec@1 {top1.avg:.3f}\t'
          'Time {batch_time.avg:.3f}\tData {data_time.avg:.3f}\t'.format(
            epoch, loss=losses, top1=top1, batch_time=batch_time, data_time=data_time))
    train_logger.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
        top1.avg, top5.avg, losses.avg, batch_time.avg, data_time.avg))
    return losses.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    test_logger.write('{0}\t{1}\n'.format(top1.avg, top5.avg))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = os.path.join(args.checkpoint_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.checkpoint_dir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed"""
    if isinstance(args.lr_decay_step, list):
        decay = len(args.lr_decay_step)
        for i, e in enumerate(args.lr_decay_step):
            if epoch < e:
                decay = i
                break
    else:
        decay = epoch // args.lr_decay_step
    lr = args.lr * (args.lr_decay ** decay)
    print('Set learning rate to {}'.format(lr))
    for group in optimizer.param_groups:
        group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def log_frequency(optimizer, epoch):
    frequency_logger = Logger(os.path.join(args.log_dir, 'freq_%d.log' % epoch))
    print('Writing {0} groups'.format(len(optimizer.frequency_list)))
    for group_id, group in enumerate(optimizer.frequency_list):
        frequency_logger.write_buf('\n')
        if isinstance(group, list):
            # print('Writing {0}-th group: {1}'.format(group_id, len(group)))
            frequency_logger.write_buf(','.join(map(str, group)))
            frequency_logger.write_buf('\n')
        else:
            dim, n = group.size()
            # print('Writing {0}-th group: {1} x {2}'.format(group_id, dim, n))
            group_cpu = group.cpu().tolist()  # group_cpu of size [dim, n]
            for i in range(dim):
                frequency_logger.write_buf(','.join(map(str, group_cpu[i])))
                frequency_logger.write_buf('\n')
    frequency_logger.close()
    print('Writing Over')


if __name__ == '__main__':
    main()

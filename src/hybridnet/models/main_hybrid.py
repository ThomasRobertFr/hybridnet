import re
import argparse
import os
import shutil
import time
import math
import logging
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from tensorboardX import SummaryWriter

from . import architectures, datasets, data, losses, ramps
from .run_context import RunContext
from .data import NO_LABEL
from .utils import *
from .ramps import Scheduler

from ..misc.tools import DotDict
from ..misc.config import merge as merge_dict
from ..misc import config as _config

config = _config.get()

args = None
best_prec1 = 0
global_step = 0
LOG = logging.getLogger('main')
tensorboard_writers = None
scheduler = None

class MeanTeacherHybrid:
    options = {
        "id": "semi-sup",
        "path": "results/semi-sup_",
        "path_tensorboard": "",

        'workers': 2,
        'checkpoint_epochs': 2,
        'evaluation_epochs': 1,
        'print_freq': 10,
        'resume': '',
        'evaluate': False,
        'pretrained': False,

        # Data
        'dataset': 'cifar10',
        'N_sup': 4000,
        'fold': 0,

        'batch_size': 128,
        'labeled_batch_size': 31,
        "exclude_unlabeled": False,

        'train_subdir': 'train+val',
        'eval_subdir': 'test',


        # Architecture
        'arch': 'cifar_shakeshake26_hybrid',
        'arch_options': {
            "type": "baseline",
            "pool_sup": "stride",
            "pool_unsup": "stride",
            "unpool_sup": "stride",
            "unpool_unsup": "stride",
            "common_enc": False,
            "short_unsup": False,
        },
        "reconstruction_wb": True,
        'ema_decay': 0.97,

        # Costs
        'consistency_type': 'mse',

        # Loss weights
        'lr': 0.05,
        'lambda_logit_dist': .01,
        'lambda_rec': 0.5,
        'lambda_rec_inter': 0.5,
        'lambda_consistency': 100.0,

        "schedules": {
            'lambda_rec' : [{"type": "exp_up", "start": 0, "end": 5}, {"type": "cosine_down", "start": 0, "end": 300}],
            'lambda_consistency': {"type": "exp_up", "start": 0, "end": 5},
            'lambda_rec_inter': {"type": "exp_up", "start": 0, "end": 2},
            "lr": {"type": "cosine_down", "start": 0, "end": 210},
        },

        # Optimization
        'optimizer': 'sgd',

        'epochs': 180,
        "start_epoch": 0,

        'nesterov': True,
        'momentum': 0.9,
        'weight_decay': 2e-4,
    }

    def __init__(self, options={}, no_logging=False):
        global args
        global tensorboard_writers
        global LOG

        args = DotDict(merge_dict(self.options, options))
        if args.labels is None:
            if args.dataset == "cifar10":
                args.labels = 'data/processed/%s/labels/%d_balanced_labels/%02d.txt' % (args.dataset, args.N_sup, args.fold)
            elif args.dataset == "svhn":
                args.labels = 'data/processed/%s/labels/%02d.txt' % (args.dataset, args.N_sup)
            else:
                args.labels = 'data/processed/%s/labels/%02d.txt' % (args.dataset, args.fold)

        # logger
        self.logger = logging.getLogger(args.id)
        self.logger.setLevel(logging.DEBUG if not no_logging else logging.ERROR)
        if not self.logger.hasHandlers():
            logger_console = logging.StreamHandler(sys.stdout)
            logger_console.setLevel(args.log_level)
            logger_console.setFormatter(
                logging.Formatter('[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)',
                                  datefmt='%Y-%m-%d %H:%M:%S'))
            self.logger.addHandler(logger_console)
        print(args)
        print(args.get_dict())
        self.logger.info("Options for this run:\n%s", yaml.dump(args.get_dict(), default_flow_style=False))
        LOG = self.logger

        tensorboard_writers = {"train": SummaryWriter(config.experiments.tensorboard_path + args.path_tensorboard + args.id+"_train"),
                                "val": SummaryWriter(config.experiments.tensorboard_path + args.path_tensorboard + args.id+"_val"),
                                "ema": SummaryWriter(config.experiments.tensorboard_path + args.path_tensorboard + args.id+"_ema")}

        with open(args.path + "meta.yml", "w") as f:
            yaml.dump(args.get_dict(), f)

    def run(self):
        context = RunContext(args.path)
        main(context)



def main(context):
    global global_step
    global best_prec1
    global scheduler

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")

    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)

    def create_model(ema=False, arch_options=args.arch_options):
        LOG.info("=> creating {pretrained}{ema}model '{arch} - {var}'".format(
            pretrained='pre-trained ' if args.pretrained else '',
            ema='EMA ' if ema else '',
            arch=args.arch,
            var=str(arch_options.get_dict())))

        # redefine gather
        def gather(self, outputs, target_device):
            dim = self.dim
            from torch.nn.parallel._functions import Gather
            def gather_map(outputs):
                out = outputs[0]
                if isinstance(out, Variable):
                    return Gather.apply(target_device, dim, *outputs)
                if isinstance(out, dict):
                    return dict([(k, Gather.apply(target_device, dim, *[each[k] for each in outputs])) for k in out.keys()])
                if out is None:
                    return None
                return type(out)(map(gather_map, zip(*outputs)))
            try:
                return gather_map(outputs)
            finally:
                gather_map = None
        nn.DataParallel.gather = gather

        model_factory = architectures.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained, num_classes=num_classes, options=arch_options)
        model = model_factory(**model_params)
        model = nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model(arch_options=args.arch_options)
    ema_opts = DotDict(args.arch_options.get_dict())
    ema_opts.type = "baseline"
    ema_model = create_model(ema=True, arch_options=ema_opts)

    #LOG.info("Model:\n\n"+str(model))
    LOG.info(parameters_string(model))

    scheduler = Scheduler(args.schedules.get_dict(),
                          {"lr": args.lr,
                           "lambda_logit_dist": args.lambda_logit_dist,
                           "lambda_consistency": args.lambda_consistency,
                           "lambda_rec": args.lambda_rec,
                           "lambda_rec_inter": args.lambda_rec_inter}, nb_epochs=args.epochs)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    weight_decay=args.weight_decay)
    else:
        assert False, "Wrong optimizer"

    # optionally resume from a checkpoint
    if args.resume:
        args.resume = args.resume.format(**args)
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    if args.evaluate:
        LOG.info("Evaluating the primary model:")
        validate(eval_loader, model, validation_log, global_step, args.start_epoch)
        LOG.info("Evaluating the EMA model:")
        validate(eval_loader, ema_model, ema_validation_log, global_step, args.start_epoch)
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        train(train_loader, model, ema_model, optimizer, epoch, training_log, tensorboard_writer=tensorboard_writers["train"])
        LOG.debug("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.debug("Evaluating the primary model:")
            prec1 = validate(eval_loader, model, validation_log, global_step, epoch + 1, tensorboard_writer=tensorboard_writers["val"])
            LOG.debug("Evaluating the EMA model:")
            ema_prec1 = validate(eval_loader, ema_model, ema_validation_log, global_step, epoch + 1, tensorboard_writer=tensorboard_writers["ema"])
            LOG.debug("--- validation in %s seconds ---" % (time.time() - start_time))
            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)

def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args):
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)
    else:
        assert False, "labels missing, TODO"

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size:
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transformation),
        batch_size=args.batch_size // 2,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)

    params = dict(model.named_parameters())
    ema_params = dict(ema_model.named_parameters())
    for ema_param_name in ema_params:
        ema_param = ema_params[ema_param_name]
        param = params[ema_param_name]
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, model, ema_model, optimizer, epoch, log, tensorboard_writer=None):
    global global_step
    global scheduler

    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    end = time.time()
    prevlogstrlen = 0
    for i, ((input, ema_input), target) in enumerate(train_loader):
        scheduler.epoch(epoch+i/len(train_loader))
        # measure data loading time
        meters.update('data_time', time.time() - end)

        # set optimizer lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = scheduler.lr

        # prepare batch
        input_var = Variable(input.cuda(non_blocking=True))
        ema_input_var = Variable(ema_input.cuda(non_blocking=True), volatile=True)
        target_var = Variable(target.cuda(non_blocking=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().item()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # forward
        ema_model_out = ema_model(ema_input_var)
        model_out = model(input_var)

        # extract outputs
        class_logit, cons_logit = model_out["y_hat"], model_out["y_cons"]
        ema_logit = ema_model_out["y_hat"]
        ema_logit = Variable(ema_logit.detach(), requires_grad=False)

        ### LOSS

        # loss - classif with y_hat
        class_loss = class_criterion(class_logit, target_var) / minibatch_size
        ema_class_loss = class_criterion(ema_logit, target_var) / minibatch_size  # for logging only

        # loss - distance between y_hat and y_cons
        res_loss = 0
        if scheduler.lambda_logit_dist >= 0:
            res_loss = residual_logit_criterion(class_logit, cons_logit) / minibatch_size
        else:
            cons_logit = class_logit

        # loss - consistency between model/y_cons and ema_moddel/y_hat
        consistency_loss = 0
        if args.lambda_consistency:
            consistency_loss = consistency_criterion(cons_logit, ema_logit) / minibatch_size

        # loss - reconstructions
        rec_loss = mse_c = mse_r = 0
        if args.arch_options.type == "ae":
            rec_loss = F.mse_loss(model_out["x_hat"], input_var)
        if args.arch_options.type == "hybrid":
            mse_c = losses.samplewise_mse(model_out["x_hat_c"], input_var)
            mse_r = losses.samplewise_mse(model_out["x_hat_r"], input_var)
            if not args.reconstruction_wb:
                rec_loss = F.mse_loss(model_out["x_hat"], input_var)
            else:
                rec_loss_c = losses.samplewise_mse(model_out["x_hat_c"] + model_out["x_hat_r"].detach(), input_var)
                rec_loss_r = losses.samplewise_mse(model_out["x_hat_c"].detach() + model_out["x_hat_r"], input_var)
                cond = (mse_c > mse_r).float()
                rec_loss = torch.mean(cond * rec_loss_c + (1 - cond) * rec_loss_r)

        # loss - intermed. reconstructions
        recint_loss = 0
        if args.arch_options.type == "ae" or args.arch_options.type == "hybrid":
            recint_loss = torch.mean(model_out["inters_c"])
            if "inters_r" in model_out:
                recint_loss = recint_loss + torch.mean(model_out["inters_r"])

        # full loss
        loss = class_loss + \
               scheduler.lambda_consistency * consistency_loss + \
               scheduler.lambda_logit_dist  * res_loss + \
               scheduler.lambda_rec         * rec_loss + \
               scheduler.lambda_rec_inter   * recint_loss
        assert not (np.isnan(loss.item()) or loss.item() > 1e8), 'Loss explosion: {}'.format(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        ### LOGGING

        # loss
        meters.update('loss/classif', class_loss.item())
        meters.update('loss/classif_ema', ema_class_loss.item())
        meters.update('loss/consistency', 0 if consistency_loss is 0 else consistency_loss.item())
        meters.update('loss/reconstruction', 0 if rec_loss is 0 else rec_loss.item())
        meters.update('loss/reconstruction_inter', 0 if recint_loss is 0 else recint_loss.item())
        meters.update('loss/reconstruction_Ec', 0 if mse_c is 0 else mse_c.mean().item())
        meters.update('loss/reconstruction_Er', 0 if mse_r is 0 else mse_r.mean().item())
        meters.update('loss/residual', 0 if res_loss is 0 else res_loss.item())
        meters.update('loss/total', loss.item())

        # perf
        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 5))
        meters.update('perf/top1', prec1, labeled_minibatch_size)
        meters.update('perf/error1', 100. - prec1, labeled_minibatch_size)
        meters.update('perf/top5', prec5, labeled_minibatch_size)
        meters.update('perf/error5', 100. - prec5, labeled_minibatch_size)

        ema_prec1, ema_prec5 = accuracy(ema_logit.data, target_var.data, topk=(1, 5))
        meters.update('perf/ema_top1', ema_prec1, labeled_minibatch_size)
        meters.update('perf/ema_error1', 100. - ema_prec1, labeled_minibatch_size)
        meters.update('perf/ema_top5', ema_prec5, labeled_minibatch_size)
        meters.update('perf/ema_error5', 100. - ema_prec5, labeled_minibatch_size)

        # print
        meters.update('batch_time', time.time() - end)
        end = time.time()

        logstr = ('Epoch {0} [{1:3d}/{2:3d} | {3:3d}/{4:3d}s]  '
                  'Loss {meters[loss/total].avg:7.4f}  /  '
                  'Cls {meters[loss/classif].avg:7.4f}  '
                  'Rec {meters[loss/reconstruction].avg:7.4f}  '
                  'Rec.int {meters[loss/reconstruction_inter].avg:7.4f}  '
                  'Cons {meters[loss/consistency].avg:7.4f}  //  '
                  'Prec@1 {meters[perf/top1].avg:5.2f}  '
                  'Prec@5 {meters[perf/top5].avg:5.2f}'.format(
            epoch, i+1, len(train_loader), int(meters["batch_time"].sum),
            int(meters["batch_time"].avg * len(train_loader)), meters=meters))
        print('\b' * prevlogstrlen, end=""); prevlogstrlen = len(logstr)
        print("\r"+logstr, end="")

        if i % args.print_freq == 0:
            LOG.debug(logstr)
            log.record(epoch + i / len(train_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })
    print()
    if tensorboard_writer is not None:
        for meter, val in meters.averages('').items():
            tensorboard_writer.add_scalar(meter, val, epoch)
        for sched, val in scheduler.schedulables.items():
            tensorboard_writer.add_scalar("sched/"+sched, val, epoch)


def validate(eval_loader, model, log, global_step, epoch, tensorboard_writer=None):
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    prevlogstrlen = 0
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        input_var = input.to(torch.device('cuda'))
        target_var = target.to(torch.device('cuda'))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().item()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output
        outputs = model(input_var)
        output1, output2 = outputs["y_hat"], outputs["y_cons"]
        softmax1, softmax2 = F.softmax(output1, dim=1), F.softmax(output2, dim=1)
        class_loss = class_criterion(output1, target_var) / minibatch_size

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 5))
        meters.update('loss/classif', class_loss.item(), labeled_minibatch_size)
        meters.update('perf/top1', prec1, labeled_minibatch_size)
        meters.update('perf/error1', 100.0 - prec1, labeled_minibatch_size)
        meters.update('perf/top5', prec5, labeled_minibatch_size)
        meters.update('perf/error5', 100.0 - prec5, labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()


        logstr = ('Eval  {0} [{1:3d}/{2:3d} | {3:3d}/{4:3d}s]  '
                      'Cls {meters[loss/classif].avg:7.4f}  '
                      'Prec@1 {meters[perf/top1].avg:5.2f}  '
                      'Prec@5 {meters[perf/top5].avg:5.2f}'.format(
                epoch, i+1, len(eval_loader), int(meters["batch_time"].sum),
                int(meters["batch_time"].avg * len(eval_loader)), meters=meters))
        print('\b' * prevlogstrlen, end="");
        prevlogstrlen = len(logstr)
        print("\r"+logstr, end="")
        if i % args.print_freq == 0:
            LOG.debug(logstr)
            log.record(epoch + i / len(eval_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })
    print()
    LOG.debug(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
          .format(top1=meters['perf/top1'], top5=meters['perf/top5']))
    log.record(epoch, {
        'step': global_step,
        **meters.values(),
        **meters.averages(),
        **meters.sums()
    })

    if tensorboard_writer is not None:
        for meter, val in meters.averages('').items():
            tensorboard_writer.add_scalar(meter, val, epoch)

    return meters['perf/top1'].avg


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format("last") #epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        LOG.info("--- checkpoint copied to %s ---" % best_path)


def accuracy(output, target, topk=(1,)):

    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum().item(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append((correct_k.mul_(100.0 / labeled_minibatch_size)).item())
    return res

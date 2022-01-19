""" Search Cell """
import os
import torch
import torch.nn as nn
import numpy as np
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler

from util import utils
import util.config as config
from util.config import config as cfg

# maybe here something with importing goes wrong when tested:
from util.verification_dataset import VerificationDataset
from ..utils.utils_callback import CallBackVerification

import util.dataset as dataset
from utils.distributed import DataloaderX

from models.search_cnn import SearchCNNController
from searchs.architect import Architect
from util.visualize import plot

# get configurations
#config = SearchConfig()

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(cfg.path, "tb"))
writer.add_text("config", config.as_markdown(), 0)

# logger
logger = utils.get_logger(os.path.join(cfg.path, "{}.log".format(cfg.name)))
config.print_params(logger.info)

def main():
    logger.info("Logger is set - training start")

    if device == "cuda":
        # set default gpu device id
        torch.cuda.set_device(0)

    # set seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

        # some optimization
        torch.backends.cudnn.benchmark = True

    # get dataset and meta info
    input_size, input_channels, n_classes, train_dataset = dataset.get_train_dataset(cfg.root, cfg.dataset)

    """
    val_data = dataset.get_dataset_without_crop(cfg.root, cfg.dataset)
    # assume that indices of train_data and val_data are the same

    # split into train val and get indices of splits
    train_idx, val_idx = dataset.get_train_val_split(train_data, cfg.dataset, 0.5)
    """

    # setup model
    criterion = nn.CrossEntropyLoss().to(device)
    header = ... # TODO: add your wanted FR header here

    model = SearchCNNController(
        input_channels,
        cfg.init_channels,
        n_classes,
        cfg.layers,
        criterion,
        header,
        cfg.n_nodes,
        cfg.stem_multiplier
    )

    model = model.to(device)

    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), cfg.w_lr, momentum=cfg.w_momentum, weight_decay=cfg.w_weight_decay)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), cfg.alpha_lr, betas=(0.5, 0.999), weight_decay=cfg.alpha_weight_decay)

    # sampler and loader for training dataset (e.g. CASIA)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True)

    callback_verification = CallBackVerification(frequent=None, rank, cfg.val_targets, cfg.rec)

    # sampler and loader for verification on validation dataset (containing only one face verification dataset)
    val_dataset = VerificationDataset(data_dir='', dataset_name='lfw', image_size=[112, 112])
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, cfg.epochs, eta_min=cfg.w_lr_min
    )

    architect = Architect(model, cfg.w_momentum, cfg.w_weight_decay)

    # training loop
    best_top1 = 0.
    for epoch in range(cfg.epochs):
        #lr = lr_scheduler.get_lr()[0]
        lr = lr_scheduler.get_last_lr()[0]

        model.print_alphas(logger)

        # training
        train(train_loader, val_loader, model, architect, w_optim, alpha_optim, lr, epoch)

        lr_scheduler.step()

        # validation
        global_step = (epoch+1) * len(train_loader)
        top1 = validate(val_loader, model, epoch, global_step)

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        plot_path = os.path.join(cfg.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        plot(genotype.normal, plot_path + "-normal", caption)
        plot(genotype.reduce, plot_path + "-reduce", caption)

        # always know which was the best cell (prevent overfitting???, kind of early stopping)
        # save
        if best_top1 <= top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        #utils.save_checkpoint(model, config.path, is_best)
        utils.save_checkpoint_search(epoch, model, w_optim, alpha_optim, top1, cfg.path, is_best)
        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))

def train(train_loader, val_loader, model, header, architect, w_optim, alpha_optim, lr, epoch):
    top1 = utils.AverageMeter()
    losses = utils.AverageMeter()

    global_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, global_step)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, val_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()

        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        #alpha_optim.step() # change position because of pytorch warning

        # phase 1. child network step (w)
        w_optim.zero_grad()

        # look SearchCNNController.loss() for details
        loss = model.loss(train_X, train_y)
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), cfg.w_grad_clip)
        w_optim.step()

        alpha_optim.step()

        prec1 = utils.accuracy(logits, trn_y, topk=(1,))
        prec1 = prec1[0]

        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)

        if step % cfg.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1) ({top1.avg:.1%})".format(
                    epoch+1, cfg.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1))

        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/top1', prec1.item(), global_step)
        global_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, cfg.epochs, top1.avg))


def validate(val_loader, model, epoch, global_step):
    val_accuracies = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        verification_results = verification_callback.get_verification_performance(model, global_step)
        mean_acc2 = sum([d['accuracy_flip'] for d in verification_results.values()]) / len(verification_results)
        val_accuracies.update(mean_acc2)

        if step % cfg.print_freq == 0 or step == len(val_loader)-1:
            logger.info("Valid: [{:2d}/{}] Prec@(1,5) ({acc.avg:.1%})".format(epoch+1, cfg.epochs, acc=val_accuracies))

    writer.add_scalar('val/top1', val_accuracies.avg, global_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, cfg.epochs, val_accuracies.avg))

    model.train()

    return val_accuracies.avg

if __name__ == "__main__":
    main()

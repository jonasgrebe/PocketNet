""" Search Cell """
import os
import torch
import torch.nn as nn
import numpy as np
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset

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
    embedding_size = 128 # as in MobileFaceNet?

    """
    ================================================================================================================
    COMMENT: We would like to perform three different kinds of experiments for an ablation study:
    validation data: used for alpha learning
    training data: used for weight learning

    (A) Identity-Joint Scenario

        (0) change nothing -> PocketNet

        (1) only change Linear Header to Margin-Penalty Header like ArcFace (Dataset = CASIA)
         -> no further changes are needed since we still optimize for classification

    (B) Identity-Disjoint Scenarios

        (2) Same changes as in (1) but in addition use different identities for training/validation
         -> this makes it necessary to use CosineEmbeddingLoss in the unrolled_backward() to calculate a loss
            on provided (img0, img1, genuine/imposter)-triples. (Dataset = CASIA)

        (3) Same changes as in (1) but in addition use different datasets for training/validation
         -> same necessities as in (2). (Datasets = CASIA for training + others for unrolled_backward and validation)

    ================================================================================================================
    """

    class LinearHeader(torch.nn.Module):
        """ Linear Header class"""

        def __init__(self, in_features, out_features):
            super(LinearHeader, self).__init__()

            self.in_features = in_features
            self.out_features = out_features

            self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)

        def forward(self, input, label):
            return self.linear(input)

    class ArcFaceHeader(torch.nn.Module):
        """ ArcFace Header class"""
        # maybe like here: https://github.com/jonasgrebe/pt-femb-face-embeddings/tree/main/femb/headers


    SCENARIO = 1 # 0 or 1 or 2 or 3

    if SCENARIO in [0, 1]:

        if SCENARIO == 0:
        # use linear header
            header = LinearHeader(embedding_size, n_classes)
        else:
            header = ArcFaceHeader(embedding_size, n_classes)

        # use uncropped images for validation
        val_dataset = dataset.get_dataset_without_crop(cfg.root, cfg.dataset)

        # assume that indices of train_data and val_data are the same
        # split into train val and get indices of splits
        train_idx, val_idx = dataset.get_train_val_split(train_dataset, cfg.dataset, 0.5)

        # loaders for train and val data
        train_loader = DataLoaderX(
            local_rank=local_rank,
            dataset=train_dataset,
            batch_size=cfg.batch_size,
            sampler=SubsetRandomSampler(train_idx),
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoaderX(
            local_rank=local_rank,
            dataset=val_dataset,
            batch_size=cfg.batch_size,
            sampler=SubsetRandomSampler(val_idx),
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )

    elif SCENARIO in [2, 3]:
        header = ArcFaceHeader(embedding_size, n_classes)

        if SCENARIO == 2:
            raise NotImplementedError("Sorry: We have to transform the val_dataset somehow into a VerificationDataset that returns (img0, img1, label), instead of (img, identity)")

            # TODO: Implement split_identity_disjoint(...) function
            train_idx, val_idx = split_identity_disjoint(train_dataset)

            # TODO: Implement Transformation from ClassificationDataset to VerificationDataset
            #       -> return (img0, img1, genuine/imposter) instead of (img, identity)
            train_dataset = Subset(train_dataset, train_idx)
            val_dataset = Subset(train_dataset, val_idx)
            # like: val_dataset = VerificationDataset(val_dataset)

        else SCENARIO == 3:
            # sampler and loader for training dataset (e.g. CASIA)

            train_dataset = train_dataset
            val_dataset = val_dataset = VerificationDataset(data_dir='', dataset_name='lfw', image_size=[112, 112])

        # sampler and loader for training dataset
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoaderX(
            local_rank=local_rank,
            dataset=train_dataset,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True)

        # sampler and loader for verification on validation dataset (containing only one face verification dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_loader = DataLoaderX(
            local_rank=local_rank,
            dataset=val_dataset,
            batch_size=cfg.batch_size,
            sampler=val_sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True)

    # setup model
    criterion = nn.CrossEntropyLoss().to(device)
    model = SearchCNNController(
        C_in=input_channels,
        C=cfg.init_channels,
        n_classes=n_classes,
        n_layers=cfg.layers,
        criterion=criterion,
        header=header,
        n_nodes=cfg.n_nodes,
        stem_multiplier=cfg.stem_multiplier
    )
    model = model.to(device)

    # weights and alphas optimizers
    w_optim = torch.optim.SGD(model.weights(), cfg.w_lr, momentum=cfg.w_momentum, weight_decay=cfg.w_weight_decay)
    alpha_optim = torch.optim.Adam(model.alphas(), cfg.alpha_lr, betas=(0.5, 0.999), weight_decay=cfg.alpha_weight_decay)

    # create verification handler
    callback_verification = CallBackVerification(frequent=None, rank, cfg.val_targets, cfg.rec)

    # create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optim, cfg.epochs, eta_min=cfg.w_lr_min)

    # create architect (and )
    architect = Architect(model, cfg.w_momentum, cfg.w_weight_decay, mode="identity-disjoint" if SCENARIO in [2, 3] else "identity-joint")


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

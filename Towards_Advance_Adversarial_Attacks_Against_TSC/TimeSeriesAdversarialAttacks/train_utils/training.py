import torch.nn.functional as F
from robustness.tools.helpers import AverageMeter, accuracy
import torch
import torch.nn as nn
import foolbox as fb
from tqdm import tqdm
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def train_step(args, model, loader, optimizer, batch_wrap=lambda x: x):
    model.train()

    for (data, target) in batch_wrap(loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        if args.distance_loss:
            if args.model != "ShapeletNet":
                raise NotImplementedError()
            output, distances = model(data, return_dist=args.distance_loss)
            loss = F.nll_loss(F.log_softmax(output), target.long())
            dist_loss = distances.mean()
            loss = loss + dist_loss * 0.1
        else:
            output = model(data)
            loss = F.nll_loss(F.log_softmax(output), target.long())
        loss.backward()
        optimizer.step()


def eval(model, loader, args, batch_wrap=lambda x: x):
    model.eval()
    losses = AverageMeter()
    if args.distance_loss and args.model == "ShapeletNet":
        dist_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for (inputs, targets) in batch_wrap(loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            N = targets.size(0)
            if args.distance_loss and args.model == "ShapeletNet":
                outputs, dists = model(inputs, args.distance_loss)
                dist_loss.update(dists.mean(), N)
            else:
                outputs = model(inputs)
            loss = F.nll_loss(F.log_softmax(outputs), targets)

            if loader.dataset.output_size < 5:
                prec1 = accuracy(outputs, targets, topk=[1])

            else:
                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                top5.update(prec5[0], N)
            losses.update(loss.item(), N)
            top1.update(prec1[0], N)

    top1_acc = top1.avg
    loss = losses.avg
    if loader.dataset.output_size < 5:
        report = {"Top1": top1_acc.item(), "Loss": loss}
    else:
        top5_acc = top5.avg
        report = {"Top1": top1_acc.item(), "Top5": top5_acc.item(), "Loss": loss}

    if args.distance_loss:
        report.update({"DistanceLoss": dist_loss.avg.item()})
    return report


def adversarial_eval(attack_fn, model, loader, epsilons, args):
    model.eval()
    print(model)
    dataset = args.dataset
    model_name = args.model
    image = dataset + "_" + model_name
    # print(dataset)
    fmodel = fb.PyTorchModel(model, bounds=(-np.inf, np.inf))
    robust_accuracies = {}
    for epsilon in epsilons:
        robust_accuracies[epsilon] = AverageMeter()

    t_loader = tqdm(loader)
    for (data, target) in t_loader:
        data, target = data.to(args.device), target.to(args.device)
        t_loader.set_description(" ".join([f"{k}: {v.avg}" for k, v in robust_accuracies.items()]))
        _, advs, success = attack_fn(fmodel, data, target, epsilons=epsilons)
        # print(model[0])
        # print(data)
        # print(target)
        ts1 = advs[1].detach().cpu().numpy()
        ts2 = data.detach().cpu().numpy()
        matplotlib.use("TkAgg")
        fig, axs = plt.subplots(ts1.shape[0], 1, figsize=(10, 80))
        for i in range(ts1.shape[0]):
            axs[i].plot(np.arange(ts1[i].shape[0]), ts1[i], label="adversarial")
            axs[i].plot(np.arange(ts2[i].shape[0]), ts2[i], label="regular")
            axs[i].legend()
        if not os.path.isdir(image):
            os.makedirs(image)
        imagepath1 = image
        imagepath1path = imagepath1 + "pgd" + ".png"
        img_path = os.path.join(image, imagepath1path)
        fig.savefig(img_path)
        print('I am in')
        robust_accuracy = 1 - success.float().mean(axis=-1)
        N = target.size(0)
        for eps, accuracy in zip(epsilons, robust_accuracy):
            robust_accuracies[eps].update(accuracy.item(), N)
    for k, v in robust_accuracies.items():
        robust_accuracies[k] = v.avg
    return robust_accuracies

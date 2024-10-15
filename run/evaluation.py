import numpy as np
import torch

from torch.nn import functional as F
from torchmetrics import Accuracy, CalibrationError
from utils import create_if_not_exists
import os

import pickle


def append_dictionary(dic1, dic2):
    """
    Extend dictionary dic1 with dic2.
    """
    for key in dic2.keys():
        if key in dic1.keys():
            dic1[key].append(dic2[key])
        else:
            dic1[key] = [dic2[key]]

def accuracy_topk(output, target, k=1):
    """Computes the topk accuracy"""
    batch_size = target.size(0)

    _, pred = torch.topk(output, k=k, dim=1, largest=True, sorted=True)

    res_total = 0
    for curr_k in range(k):
      curr_ind = pred[:,curr_k]
      num_eq = torch.eq(curr_ind, target).sum()
      acc = num_eq/len(output)
      res_total += acc
    return res_total*100

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate_all(model, dataloader, accelerator, args, sample=True, num_classes=None):
    """
    Evaluates the **acc, ece, nll** of the model for given dataset.

    Args:
        model: the model to be evaluated
        dataloader: the dataset to evaluate the model on
        kwargs: optional arguments
    Returns:
        acc: accuracy of the model evaluated on the dataloader
    """
    status = model.net.training
    model.net.eval()

    nlls = AverageMeter()
    if num_classes is None:
        num_classes = args.outdim
    metric_kwargs = {"task": "multiclass", "num_classes": num_classes}
    acc_metric = Accuracy(**metric_kwargs).to(accelerator.device)
    ece_metric = CalibrationError(**metric_kwargs, n_bins = args.num_bins).to(accelerator.device)
    briers = AverageMeter()

    samples_seen = 0
    for step, batch in enumerate(dataloader):
        with torch.no_grad() and torch.inference_mode():
            if args.dataset_type == 'mcdataset':
                _, labels, _ = batch
                logits = model(batch, sample = not args.bayes_inference_notsample).detach()
            else:
                logits = model(batch, sample = not args.bayes_inference_notsample).detach()
                labels = batch["labels"]
            logits, labels = accelerator.gather([logits, labels])
            if accelerator.num_processes > 1:
                if step == len(dataloader) - 1:
                    labels = labels[: len(dataloader.dataset) - samples_seen]
                    logits = logits[: len(dataloader.dataset) - samples_seen]
                else:
                    samples_seen += labels.shape[0]
            # loss_func = 
            # loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
            # nll = loss_func(logits, labels)
            # nlls.update(nll)

            if (not args.bayes_inference_notsample and args.model.startswith('blob')) or args.model.startswith('deepensemble') or args.model.startswith('mcdropout'):
                probs = torch.softmax(logits, dim=-1).mean(dim=1)
                std = torch.softmax(logits, dim=-1).std(dim=1).mean()
            else:
                probs = torch.softmax(logits, dim=-1)
                std = 0

            acc_metric(probs, labels)
            ece_metric(probs, labels)
            loss_func = torch.nn.NLLLoss(reduction="mean")
            nll = loss_func(torch.log(probs), labels)
            nlls.update(nll)

            brier = (probs - F.one_hot(labels, num_classes=logits.size(-1))).pow(2).sum(dim=-1).mean()
            briers.update(brier)
            
    acc = acc_metric.compute().item()
    ece = ece_metric.compute().item()
    nll = nlls.avg
    brier = briers.avg
    model.net.train(status)
        
    return acc, ece, nll, std, brier

def logit_entropy(probs):
    return (-torch.sum(probs * torch.log(probs), dim=1)).cpu().numpy()

def max_softmax(probs):
    return (1 - probs.max(dim=1)[0]).cpu().numpy()

def logit_std(probs):
    return (probs.std(dim=1)).cpu().numpy()

def evaluate_ood_detection(model, dataset, ood_ori_dataset, dataloader, ood_ori_dataloader, accelerator, args, sample=True, num_classes=None):
    """
    Evaluates the **acc, ece, nll** of the model for given dataset.

    Args:
        model: the model to be evaluated
        dataloader: the dataset to evaluate the model on
        kwargs: optional arguments
    Returns:
        acc: accuracy of the model evaluated on the dataloader
    """
    status = model.net.training
    model.net.eval()

    model.tokenizer = ood_ori_dataset.tokenizer
    model.net.module.target_ids = ood_ori_dataset.target_ids.squeeze(-1)
    model.net.module.model.target_ids = ood_ori_dataset.target_ids.squeeze(-1)

    if num_classes is None:
        num_classes = args.outdim
    samples_seen = 0
    id_prob_list = np.array([])
    for step, batch in enumerate(ood_ori_dataloader):
        with torch.no_grad() and torch.inference_mode():
            if args.dataset_type == 'mcdataset':
                _, labels, _ = batch
                logits = model(batch, sample = not args.bayes_inference_notsample).detach()
            else:
                logits = model(batch, sample = not args.bayes_inference_notsample).detach()
                labels = batch["labels"]
            logits, labels = accelerator.gather([logits, labels])
            if accelerator.num_processes > 1:
                if step == len(ood_ori_dataloader) - 1:
                    labels = labels[: len(ood_ori_dataloader.dataset) - samples_seen]
                    logits = logits[: len(ood_ori_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += labels.shape[0]
            # loss_func = 
            # loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
            # nll = loss_func(logits, labels)
            # nlls.update(nll)

            if not args.bayes_inference_notsample and args.model.startswith('blob'):
                pre_mean_probs = torch.softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1).mean(dim=1)
                std = torch.softmax(logits, dim=-1).std(dim=1).mean()
                logits = logits.mean(dim=1)
            else:
                probs = torch.softmax(logits, dim=-1)
                std = 0

            if args.ood_detection_method == "max-softmax":
                id_probs = max_softmax(probs)
            elif args.ood_detection_method == "logits-std":
                id_probs = logit_std(logits)
            elif args.ood_detection_method == "logits-entropy":
                id_probs = logit_entropy(probs)
            else:
                raise NotImplementedError(f"OOD detection method {args.ood_detection_method} not implemented.")
            id_prob_list = np.append(id_prob_list, id_probs)
    id_label_list = np.zeros_like(id_prob_list)

    model.tokenizer = dataset.tokenizer
    model.net.module.target_ids = dataset.target_ids.squeeze(-1)
    model.net.module.model.target_ids = dataset.target_ids.squeeze(-1)

    ood_prob_list = np.array([])
    samples_seen = 0
    nlls = AverageMeter()
    if num_classes is None:
        num_classes = args.outdim
    metric_kwargs = {"task": "multiclass", "num_classes": num_classes}
    acc_metric = Accuracy(**metric_kwargs).to(accelerator.device)
    ece_metric = CalibrationError(**metric_kwargs, n_bins = args.num_bins).to(accelerator.device)
    briers = AverageMeter()
    for step, batch in enumerate(dataloader):
        with torch.no_grad() and torch.inference_mode():
            if args.dataset_type == 'mcdataset':
                _, labels, _ = batch
                logits = model(batch, sample = not args.bayes_inference_notsample).detach()
            else:
                logits = model(batch, sample = not args.bayes_inference_notsample).detach()
                labels = batch["labels"]
            logits, labels = accelerator.gather([logits, labels])
            if accelerator.num_processes > 1:
                if step == len(dataloader) - 1:
                    labels = labels[: len(dataloader.dataset) - samples_seen]
                    logits = logits[: len(dataloader.dataset) - samples_seen]
                else:
                    samples_seen += labels.shape[0]

            if not args.bayes_inference_notsample and args.model.startswith('blob'):
                pre_mean_probs = torch.softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1).mean(dim=1)
                std = torch.softmax(logits, dim=-1).std(dim=1).mean()
                logits = logits.mean(dim=1)
            else:
                probs = torch.softmax(logits, dim=-1)
                std = 0

            acc_metric(probs, labels)
            ece_metric(probs, labels)
            loss_func = torch.nn.NLLLoss(reduction="mean")
            nll = loss_func(torch.log(probs), labels)
            nlls.update(nll)

            brier = (probs - F.one_hot(labels, num_classes=logits.size(-1))).pow(2).sum(dim=-1).mean()
            briers.update(brier)

            if args.ood_detection_method == "max-softmax":
                ood_probs = max_softmax(probs)
            elif args.ood_detection_method == "logits-std":
                ood_probs = logit_std(logits)
            elif args.ood_detection_method == "logits-entropy":
                ood_probs = logit_entropy(probs)
            else:
                raise NotImplementedError(f"OOD detection method {args.ood_detection_method} not implemented.")
            ood_prob_list = np.append(ood_prob_list, ood_probs)

    acc = acc_metric.compute().item()
    ece = ece_metric.compute().item()
    nll = nlls.avg
    brier = briers.avg
            
    ood_label_list = np.ones_like(ood_prob_list)
    labels = np.concatenate((id_label_list, ood_label_list))
    probs = np.concatenate((id_prob_list, ood_prob_list))

    # log the scores
    create_if_not_exists('log-ood-detection')
    with open(os.path.join('log-ood-detection', f'{args.model}-{args.dataset}-{args.ood_detection_method}-seed{args.seed}.pkl'), 'wb') as f:
        to_dump = {"labels": labels, "scores": probs}
        pickle.dump(to_dump, f)

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    best_threshold_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_threshold_index]

    print("Best Threshold:", best_threshold)

    predictions = [1 if prob >= best_threshold else 0 for prob in probs]
    total_samples = len(labels)
    correct_predictions = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    acc_ood = correct_predictions / total_samples
            
    model.net.train(status)
        
    return acc, ece, nll, brier, acc_ood, roc_auc



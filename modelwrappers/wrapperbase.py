# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from contextlib import suppress
from tqdm import tqdm
import math
import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from torchmetrics import Accuracy, CalibrationError

from transformers import PreTrainedModel
from peft import PeftModel
from peft.config import PeftConfig

from utils import create_if_not_exists

optimizer_dict = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW,
}


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def accuracy_topk(output, target, k=1):
    """Computes the topk accuracy"""
    batch_size = target.size(0)

    _, pred = torch.topk(output, k=k, dim=1, largest=True, sorted=True)

    res_total = 0
    for curr_k in range(k):
        curr_ind = pred[:, curr_k]
        num_eq = torch.eq(curr_ind, target).sum()
        acc = num_eq / len(output)
        res_total += acc
    return res_total * 100


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


class WrapperBase(PeftModel):
    """
    Base ModelWrapper for this project.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        args,
        accelerator,
        adapter_name: str = "default",
    ):
        """Initializes the model wrapper.

        Args:
            model (PreTrainedModel): The pretrained model to wrap.
            peft_config (PeftConfig): The configuration for parameter-efficient fine-tuning (PEFT).
            args (argparse.Namespace): Arguments with configuration for training.
            accelerator (Accelerator): The accelerator to handle multi-GPU or mixed precision.
            adapter_name (str, optional): The name of the adapter. Defaults to "default".
        """
        super().__init__(model, peft_config, adapter_name)

        self.loss = F.nll_loss
        self.args = args
        self.accelerator = accelerator
        self.target_ids = None

        self.batch_size = args.batch_size
        self.num_epochs = args.n_epochs
        self.num_training_steps = args.max_train_steps
        self.step = 0
        self.num_classes = args.outdim
        self.eval_n_samples = 1

        if args.max_train_steps == 0:
            num_training_steps = args.num_samples * args.n_epochs // args.batch_size
        else:
            num_training_steps = args.max_train_steps
        warmup_steps = num_training_steps * args.warmup_ratio
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                # set weight_decay
                "weight_decay": args.opt_wd,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if args.opt == "adamw" or args.opt == "adam":
            self.opt = optimizer_dict[args.opt](
                optimizer_grouped_parameters,
                lr=args.lr,
                eps=args.adam_epsilon,
                weight_decay=args.opt_wd,
            )
        else:
            self.opt = optimizer_dict[args.opt](
                optimizer_grouped_parameters, lr=args.lr, weight_decay=args.opt_wd
            )
        self.scheduler = get_linear_schedule_with_warmup(
            self.opt, warmup_steps, num_training_steps
        )

    def forward_logits(self, *args, **kwargs) -> torch.Tensor:
        """Defines the forward pass for computing logits. This method is not implemented 
        in the base class and needs to be implemented in subclasses.
        
        Returns:
            torch.Tensor: The logits predicted by the model.
        """
        raise NotImplementedError("Forward not implemented.")

    def fit(self, train_loader, eval_loader):
        """Fits the model using the training data and evaluates it periodically.
        
        Args:
            train_loader (DataLoader): The training data loader.
            eval_loader (DataLoader): The evaluation data loader.
        """
        nll_losses = AverageMeter()
        accs = AverageMeter()
        samples_seen = 0
        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {self.args.epoch+1}/{self.args.n_epochs}",
            leave=False,
        ) as pbar:
            for i, batch in enumerate(train_loader):
                if self.args.dataset_type == "mcdataset":
                    _, golds, _ = batch
                elif self.args.dataset_type == "bertds":
                    golds = batch["labels"]
                else:
                    raise NotImplementedError(
                        f"Dataset type {self.args.dataset_type} not implemented."
                    )
                logits = self.forward_logits(batch).mean(1)
                output = torch.log_softmax(logits, dim=1)
                nll = self.loss(output, golds, reduction="mean")

                self.accelerator.backward(nll)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                acc = accuracy_topk(output.data, golds)
                acc, nll_loss = acc.item(), nll.detach().cpu().numpy()

                if self.args.dataset_type == "mcdataset":
                    _, classes, _ = batch
                    references = self.accelerator.gather(classes)
                else:
                    references = self.accelerator.gather(batch["labels"])
                if self.accelerator.num_processes > 1:
                    if i == len(train_loader) - 1:
                        references = references[
                            : len(train_loader.dataset) - samples_seen
                        ]
                    else:
                        samples_seen += references.shape[0]
                len_batch = references.shape[0]
                nll_losses.update(nll_loss, len_batch)
                accs.update(acc, len_batch)

                assert not math.isnan(nll_loss)
                if self.accelerator.is_local_main_process:
                    if self.wandb_logger is not None:
                        self.wandb_logger.log(
                            {
                                "train_acc": accs.avg,
                                "train_nll_loss": nll_losses.avg,
                                "lr": self.opt.param_groups[0]["lr"],
                            }
                        )

                self.step += self.accelerator.num_processes
                pbar.update(1)
                if self.step >= self.args.eval_per_steps:
                    self.step -= self.args.eval_per_steps
                    self.evaluate(eval_loader)

    def evaluate(self, eval_loader):
        """Evaluates the model using the evaluation data.
    
        Args:
            eval_loader (DataLoader): The evaluation data loader.
        
        Returns:
            tuple: The evaluation results: accuracy, ECE (Expected Calibration Error), 
                negative log-likelihood (NLL), and Brier score.
        """
        self.eval()
        status = self.training
        nlls = AverageMeter()
        metric_kwargs = {"task": "multiclass", "num_classes": self.num_classes}
        acc_metric = Accuracy(**metric_kwargs).to(self.accelerator.device)
        ece_metric = CalibrationError(**metric_kwargs, n_bins=self.args.num_bins).to(
            self.accelerator.device
        )
        briers = AverageMeter()

        samples_seen = 0
        for step, batch in enumerate(eval_loader):
            with torch.no_grad() and torch.inference_mode():
                logits = self.forward_logits(
                    batch, sample=True, n_samples=self.eval_n_samples
                ).detach()
                if self.args.dataset_type == "mcdataset":
                    _, labels, _ = batch
                else:
                    labels = batch["labels"]
                logits, labels = self.accelerator.gather([logits, labels])
                if self.accelerator.num_processes > 1:
                    if step == len(eval_loader) - 1:
                        labels = labels[: len(eval_loader.dataset) - samples_seen]
                        logits = logits[: len(eval_loader.dataset) - samples_seen]
                    else:
                        samples_seen += labels.shape[0]
                probs = torch.softmax(logits, dim=-1).mean(dim=1)
                if self.eval_n_samples > 1:
                    std = torch.softmax(logits, dim=-1).std(dim=1).mean()
                else:
                    std = 0

                acc_metric(probs, labels)
                ece_metric(probs, labels)
                nll = self.loss(torch.log(probs), labels, reduction="mean")
                if torch.isnan(nll):
                    if self.accelerator.is_local_main_process:
                        print("nll:", nll)
                        print("probs:", probs)
                        print("logits:", logits)
                        exit()
                nlls.update(nll)

                brier = (
                    (probs - F.one_hot(labels, num_classes=logits.size(-1)))
                    .pow(2)
                    .sum(dim=-1)
                    .mean()
                )
                briers.update(brier)

        val_acc = acc_metric.compute().item()
        val_ece = ece_metric.compute().item()
        val_nll = nlls.avg
        val_brier = briers.avg
        self.train(status)

        if self.accelerator.is_local_main_process:
            if self.wandb_logger is not None:
                self.wandb_logger.log(
                    {
                        "val_acc": val_acc,
                        "val_ece": val_ece,
                        "val_nll": val_nll,
                        "std": std,
                        "val_brier": val_brier,
                    }
                )
        return val_acc, val_ece, val_nll, val_brier

    def fit_evaluate(self):
        """Performs the fitting and evaluation process, saving the results to checkpoints 
        and logging them to the WandB logger.
        """
        if self.accelerator.is_local_main_process:
            save_folder = f"checkpoints/{self.args.modelwrapper}/{self.args.model}/{self.args.dataset}/{self.args.log_path}"
            create_if_not_exists(save_folder)
            logging.basicConfig(
                format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
                level=logging.INFO,
                filename=save_folder + "/log.txt",
            )
        with tqdm(
            total=self.args.n_epochs, desc=f"Total Training Epochs", leave=True
        ) as pbar:
            for epoch in range(self.args.n_epochs):
                if self.args.early_stop_steps > 0 and epoch >= self.earlystop_n_epochs:
                    break
                self.args.epoch = epoch
                self.fit(self.train_loader, self.test_loader)
                pbar.update(1)

        if hasattr(self.args, "bayes_eval_n_samples_final"):
            self.eval_n_samples = self.args.bayes_eval_n_samples_final

        val_acc, val_ece, val_nll, val_brier = self.evaluate(self.test_loader)
        logging.info(
            f"val_acc: {val_acc}, val_ece: {val_ece}, val_nll: {val_nll}, val_brier: {val_brier}"
        )
        if self.accelerator.is_local_main_process:
            if self.wandb_logger is not None:
                self.wandb_logger.log(
                    {
                        "final_val_acc": val_acc,
                        "final_val_ece": val_ece,
                        "final_val_nll": val_nll,
                        "final_val_brier": val_brier,
                    }
                )

    def prepare_for_fit_evaluate(self, dataset, wandb_logger=None):
        """Prepares the model and data loaders for training and evaluation.
    
        Args:
            dataset (Dataset): The dataset object containing train and test data loaders.
            wandb_logger (optional): The Weights & Biases logger for tracking experiments.
        """
        self.wandb_logger = wandb_logger
        train_loader, test_loader = dataset.train_dataloader, dataset.test_dataloader
        if self.args.testing_set == "train_val":
            val_loader = dataset.val_dataloader
            val_loader = self.accelerator.prepare(val_loader)
            self.val_loader = val_loader

        if self.args.dataset_type == "mcdataset":
            self.tokenizer = dataset.tokenizer
            self.target_ids = dataset.target_ids.squeeze(-1)

        num_update_steps_per_epoch = math.ceil(len(train_loader))
        if self.args.max_train_steps == 0:
            self.args.max_train_steps = self.args.n_epochs * num_update_steps_per_epoch
        self.args.n_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch
        )
        if self.args.early_stop_steps > 0:
            self.earlystop_n_epochs = math.ceil(
                self.args.early_stop_steps / num_update_steps_per_epoch
            )
        else:
            self.earlystop_n_epochs = 0
        if self.accelerator.is_local_main_process:
            print("len(train_loader):", len(train_loader))
            print("num of epochs:", self.args.n_epochs)
        self.step = 0

        self.base_model, self.opt, train_loader, test_loader, self.scheduler = (
            self.accelerator.prepare(
                self.base_model, self.opt, train_loader, test_loader, self.scheduler
            )
        )

        self.train_loader = train_loader
        self.test_loader = test_loader

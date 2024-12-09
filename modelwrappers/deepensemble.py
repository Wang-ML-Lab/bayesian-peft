# Link to the baseline paper: https://arxiv.org/abs/1612.01474.

import torch
from .wrapperbase import WrapperBase
from tqdm import tqdm
import math

from .wrapperbase import get_linear_schedule_with_warmup, optimizer_dict

from utils.args import add_management_args, add_experiment_args, ArgumentParser
from run.evaluation import *
from transformers import PreTrainedModel
from peft.config import PeftConfig


## Model Specific Argument Parsing
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Deep Ensemble.")
    add_management_args(parser)
    add_experiment_args(parser)

    # DeepEnsemble-specific arguments.
    parser.add_argument("--ensemble-n", type=int, default=3)
    return parser


class DeepEnsemble(WrapperBase):
    """Deep Ensemble model."""

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        args,
        accelerator,
        adapter_name: str = "default",
    ):
        super().__init__(model, peft_config, args, accelerator, adapter_name)
        self.ensemble_n = args.ensemble_n

        # create a couple of copies for the LoRAs in the memory
        ## get the peft configs and use it to create n-1 number of peft modules.
        self.opts, self.schedulers = [self.opt], [self.scheduler]
        for i in range(1, self.ensemble_n):
            self.add_adapter(peft_config=peft_config, adapter_name=f"adapter_{i}")
            opt, scheduler = self.set_additional_opts(args)
            self.opts.append(opt)
            self.schedulers.append(scheduler)
        if args.load_lora_path is not None:
            for i in range(1, self.ensemble_n):
                self.load_adapter(args.load_lora_path, adapter_name=f"adapter_{i}")
        self.adapter_names = list(self.peft_config.keys())
        print(self.adapter_names)

    def fit(self, train_loader, eval_loader):
        def single_train_iter(self, batch, opt, scheduler):
            """original MLE training script."""
            if self.args.dataset_type == "mcdataset":
                prompts, classes, _ = batch
                inputs = prompts.to(self.accelerator.device)
                golds = classes.to(self.accelerator.device)
                output = self.base_model(**inputs)
                logits = output.logits[:, -1, self.target_ids]
            elif self.args.dataset_type == "bertds":
                batch_in_token_ids = batch["input_ids"].to(self.accelerator.device)
                attention_mask = batch["attention_mask"].to(self.accelerator.device)
                golds = batch["labels"].to(self.accelerator.device)
                logits = self.base_model(
                    **{
                        "input_ids": batch_in_token_ids,
                        "attention_mask": attention_mask,
                    }
                )
            else:
                raise NotImplementedError(
                    f"Dataset type {self.args.dataset_type} not implemented."
                )
            outputs = torch.log_softmax(logits, dim=-1)
            nll = self.loss(outputs, golds, reduction="mean")

            opt.zero_grad()
            self.accelerator.backward(nll)
            opt.step()
            scheduler.step()
            acc = accuracy_topk(outputs.data, golds)
            return nll.item(), acc.item()

        nll_losses = AverageMeter()
        accs = AverageMeter()
        samples_seen = 0
        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {self.args.epoch+1}/{self.args.n_epochs}",
            leave=False,
        ) as pbar:
            for i, batch in enumerate(train_loader):
                # train each of the adapter in the ensemble.
                for i, adapter in enumerate(self.adapter_names):
                    opt, scheduler = self.opts[i], self.schedulers[i]
                    self.set_adapter(adapter)
                    nll_loss, acc = single_train_iter(self, batch, opt, scheduler)

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

    def forward_logits(self, batch, **kwargs) -> torch.Tensor:
        if self.args.dataset_type == "mcdataset":
            inputs, _, _ = batch
            # ensemble the results.
            logits_list = []
            for adapter in self.adapter_names:
                self.set_adapter(adapter)
                output = self.base_model(**inputs)
                logits = output.logits[:, -1, self.target_ids]
                logits_list.append(logits)
        else:
            # ensemble the results.
            logits_list = []
            for adapter in self.adapter_names:
                self.set_adapter(adapter)
                logits_list.append(self.base_model(**batch).logits)
            # return the result.
        return torch.stack(logits_list, dim=1)

    def evaluate(self, eval_loader):
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
                if self.args.dataset_type == "mcdataset":
                    _, labels, _ = batch
                    logits = self.forward_logits(batch).detach()
                else:
                    logits = self.forward_logits(batch).detach()
                    labels = batch["labels"]
                logits, labels = self.accelerator.gather([logits, labels])
                if self.accelerator.num_processes > 1:
                    if step == len(eval_loader) - 1:
                        labels = labels[: len(eval_loader.dataset) - samples_seen]
                        logits = logits[: len(eval_loader.dataset) - samples_seen]
                    else:
                        samples_seen += labels.shape[0]
                # compute the mean of logits and then compute the softmax
                probs = torch.softmax(logits.mean(dim=1), dim=-1)
                std = logits.std(dim=1).mean()

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

    def set_adapter(self, adapter):
        """To deal with DDP-wrapped model."""
        if isinstance(self.base_model, torch.nn.parallel.DistributedDataParallel):
            self.base_model.module.set_adapter(adapter)
        else:
            self.base_model.set_adapter(adapter)

    def set_additional_opts(self, args):
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
            opt = optimizer_dict[args.opt](
                optimizer_grouped_parameters,
                lr=args.lr,
                eps=args.adam_epsilon,
                weight_decay=args.opt_wd,
            )
        else:
            opt = optimizer_dict[args.opt](
                optimizer_grouped_parameters, lr=args.lr, weight_decay=args.opt_wd
            )
        scheduler = get_linear_schedule_with_warmup(
            opt, warmup_steps, num_training_steps
        )

        return opt, scheduler

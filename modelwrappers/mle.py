import torch
import math
from tqdm import tqdm

from .wrapperbase import WrapperBase
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from run.evaluation import *

from transformers import PreTrainedModel
from peft.config import PeftConfig

## Model Specific Argument Parsing
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='standard training, MLE')
    add_management_args(parser)
    add_experiment_args(parser)

    return parser


class MLE(WrapperBase):
    """MLE model."""
    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, args, accelerator, adapter_name: str = "default"):
        super().__init__(model, peft_config, args, accelerator, adapter_name)
        
    def forward_logits(self, batch, sample=True, n_samples=1, **kwargs) -> torch.Tensor:
        if self.args.dataset_type == 'mcdataset':
            inputs, _, _ = batch
            output = self.base_model(**inputs)
            logits = output.logits[:, -1, self.target_ids]
            return logits.unsqueeze(1)
        else:
            res = self.base_model(**batch.logits)
            return res.unsqueeze(1)
    
    def fit(self, train_loader, eval_loader):
        nll_losses = AverageMeter()
        accs = AverageMeter()   
        samples_seen = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {self.args.epoch+1}/{self.args.n_epochs}", leave=False) as pbar:
            for i, batch in enumerate(train_loader): 
                if self.args.dataset_type == 'mcdataset':
                    _, golds, _ = batch
                elif self.args.dataset_type == 'bertds':
                    golds = batch['labels']
                else:
                    raise NotImplementedError(f"Dataset type {self.args.dataset_type} not implemented.")
                logits = self.forward_logits(batch).mean(1)
                output = torch.log_softmax(logits, dim=1)
                nll = self.loss(output, golds, reduction='mean')

                self.accelerator.backward(nll)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                acc = accuracy_topk(output.data, golds)
                acc, nll_loss = acc.item(), nll.detach().cpu().numpy()

                if self.args.dataset_type == 'mcdataset':
                    _, classes, _ = batch
                    references = self.accelerator.gather(classes)
                else:
                    references = self.accelerator.gather(batch["labels"])
                if self.accelerator.num_processes > 1:
                    if i == len(train_loader) - 1:
                        references = references[: len(train_loader.dataset) - samples_seen]
                    else:
                        samples_seen += references.shape[0]
                len_batch = references.shape[0]
                nll_losses.update(nll_loss, len_batch)
                accs.update(acc, len_batch)
                
                assert not math.isnan(nll_loss)
                if self.accelerator.is_local_main_process:
                    if self.wandb_logger is not None:
                        self.wandb_logger.log({
                                'train_acc': accs.avg, 
                                'train_nll_loss': nll_losses.avg, 
                                'lr': self.opt.param_groups[0]['lr'],
                            })
                
                self.step += self.accelerator.num_processes
                pbar.update(1)
                if self.step >= self.args.eval_per_steps:
                    self.step -= self.args.eval_per_steps
                    self.evaluate(eval_loader)

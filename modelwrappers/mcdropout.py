# Link to the baseline paper: https://arxiv.org/abs/1506.02142.

import torch
from .wrapperbase import WrapperBase
from tqdm import tqdm
import math

from utils.args import add_management_args, add_experiment_args, ArgumentParser
from run.evaluation import *
from transformers import PreTrainedModel
from peft.config import PeftConfig


## Model Specific Argument Parsing
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Monte-Carlo Dropout, MC Dropout.")
    add_management_args(parser)
    add_experiment_args(parser)

    parser.add_argument(
        "--bayes-eval-n-samples",
        type=int,
        default=1,
        help="Number of samples to use for evaluation during training.",
    )
    parser.add_argument(
        "--bayes-eval-n-samples-final",
        type=int,
        default=10,
        help="Number of samples to use for evaluation.",
    )
    return parser


class MCDropout(WrapperBase):
    """MC Dropout Model"""

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        args,
        accelerator,
        adapter_name: str = "default",
    ):
        super().__init__(model, peft_config, args, accelerator, adapter_name)
        if args.load_lora_path is not None:
            self.load_adapter(args.load_lora_path, adapter_name)
        self.train_n_samples = 1
        self.eval_n_samples = self.args.bayes_eval_n_samples
        self.dropouts = []
        for _, module in self.named_modules():
            if isinstance(module, torch.nn.modules.dropout.Dropout):
                self.dropouts.append(module)

    def forward_logits(self, batch, sample=True, n_samples=1, **kwargs) -> torch.Tensor:
        if self.args.dataset_type == "mcdataset":
            inputs, _, _ = batch
            if not sample:
                res = self.base_model(**inputs)
                return res
            else:
                # no matter in what model, eval/train,
                # brute-forcely set the training flag to True and then set it back.
                old_training = self.dropouts[0].training
                for dropout in self.dropouts:
                    dropout.training = True
                # Then do the sampling.
                logits_list = []
                for _ in range(n_samples):
                    output = self.base_model(**inputs)
                    logits = output.logits[:, -1, self.target_ids]
                    logits_list.append(logits)
                # reset the training flag.
                for dropout in self.dropouts:
                    dropout.training = old_training
                # return the result.
                return torch.stack(logits_list, dim=1)
        else:
            if not sample:
                self.sample(self.base_model, False)
                res = self.base_model(**batch)
                self.sample(self.base_model, True)
                return res
            else:
                # no matter in what model, eval/train,
                # brute-forcely set the training flag to True and then set it back.
                old_training = self.dropouts[0].training
                for dropout in self.dropouts:
                    dropout.training = True
                # Then do the sampling.
                logits_list = []
                for _ in range(n_samples):
                    logits_list.append(self.base_model(**batch).logits)
                # reset the training flag.
                for dropout in self.dropouts:
                    dropout.training = old_training
                # return the result.
                return torch.stack(logits_list, dim=1)

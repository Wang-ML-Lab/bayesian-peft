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
    parser = ArgumentParser(description="standard training, MLE")
    add_management_args(parser)
    add_experiment_args(parser)

    return parser


class MLE(WrapperBase):
    """MLE model."""

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

    def forward_logits(self, batch, sample=True, n_samples=1, **kwargs) -> torch.Tensor:
        if self.args.dataset_type == "mcdataset":
            inputs, _, _ = batch
            output = self.base_model(**inputs)
            logits = output.logits[:, -1, self.target_ids]
            return logits.unsqueeze(1)
        else:
            res = self.base_model(**batch.logits)
            return res.unsqueeze(1)

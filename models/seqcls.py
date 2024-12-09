import torch.nn as nn
from run import get_modelwrapper

from transformers import AutoModelForSequenceClassification
from peft import (
    get_peft_model,
    LoraConfig,
    PeftModel,
    PeftConfig,
)


class SeqCls(nn.Module):
    def __init__(self, args, accelerator=None, tokenizer=None, **kwargs) -> None:
        super().__init__()
        self.args = args
        if accelerator is not None:
            accelerator.wait_for_everyone()
            
        if args.load_model_path is not None:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.load_model_path, num_labels=args.outdim
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model, num_labels=args.outdim
            )
        if args.apply_classhead_lora:
            target_modules = [
                "query",
                "value",
                "classifier.dense",
                "classifier.out_proj",
            ]
        else:
            target_modules = ["query", "value"]

        peft_config = LoraConfig(
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        self.model = get_peft_model(model, peft_config)
        self.peft_config = peft_config


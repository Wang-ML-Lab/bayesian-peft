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
        if args.load_checkpoint:
            print('=====================')
            load_path = f'checkpoints/{args.dataset}/{args.model}/{args.model}/{args.load_model_path}'
            print('Loading model from: ', load_path)
            peft_config = PeftConfig.from_pretrained(load_path, is_trainable=True)
            model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
            self.model = PeftModel.from_pretrained(model, load_path, is_trainable=True)
            modelwrapper = get_modelwrapper(args.modelwrapper)
            self.model = modelwrapper(self.model, peft_config, args, accelerator, adapter_name="default")
            self.model.print_trainable_parameters()
            print('Model loaded successfully')
            print('=====================')
        else:
            if args.load_model_path is not None:
                model = AutoModelForSequenceClassification.from_pretrained(args.load_model_path, num_labels=args.outdim)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.outdim)
            if args.apply_classhead_lora:
                target_modules=['query', 'value', 'classifier.dense', 'classifier.out_proj']
            else:
                target_modules=['query', 'value']
                
            peft_config = LoraConfig(inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=target_modules)
            self.model = get_peft_model(model, peft_config)
            modelwrapper = get_modelwrapper(args.modelwrapper)
            self.model = modelwrapper(self.model, peft_config, args, accelerator, adapter_name="default")
            self.model.print_trainable_parameters()
        

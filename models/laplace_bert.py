import torch
import torch.nn as nn
import math

from laplace_bayeslib import Laplace
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PeftModel,
    PeftConfig
)
from transformers.models.roberta import RobertaForSequenceClassification, RobertaConfig

class laplace_bert(nn.Module):
    def __init__(self, args, accelerator=None, tokenizer=None, **kwargs) -> None:
        super().__init__()
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if args.model.startswith('roberta'):
            if args.load_model_path is not None:
                # accelerator.wait_for_everyone()
                # if accelerator.is_main_process:
                # Load the pretrained checkpoint first
                print('=====================')
                load_path = f'checkpoints/{args.dataset}/{args.model}/{args.model}/{args.load_model_path}'
                print('Loading model from: ', load_path)
                peft_config = PeftConfig.from_pretrained(load_path, is_trainable=True)
                model = RobertaForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
                self.model = PeftModel.from_pretrained(model, load_path, is_trainable=True)
                self.model.print_trainable_parameters()
                print('Model loaded successfully')
                print('=====================')
            else:
                model = RobertaForSequenceClassification.from_pretrained(args.model, num_labels=args.outdim)
                # target_modules=['query', 'value', 'classifier.dense', 'classifier.out_proj']
                target_modules=['query', 'value']
                peft_config = LoraConfig(inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=target_modules)
                self.model = get_peft_model(model, peft_config)

        else:
            raise NotImplementedError("Unsupported backbone type")

    
    def forward(self, batch, returnt='logits'):
        # sample is useless for normal lora_roberta
        batch_in_token_ids = batch[0]
        attention_mask = batch[1]
        if returnt == 'loss':
            labels = batch[2]
        else:
            labels = None

        output = self.model(input_ids = batch_in_token_ids, attention_mask = attention_mask, labels = labels)
        logits = output.logits

        if returnt == 'logits':
            return logits
        elif returnt == 'prob':
            return torch.softmax(logits, dim=1)
        elif returnt == 'log_prob':
            return torch.log_softmax(logits, dim=1)
        elif returnt == 'loss':
            return output.loss
        elif returnt == 'all':
            return logits, torch.softmax(logits, dim=1)
        else:
            return NotImplementedError("Unsupported return type")
        
    def reload(self, args, accelerator):
        super().__init__()
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print('=====================')
            load_path = f'checkpoints/{args.dataset}/{args.backbone}/{args.model}/{args.load_model_path}'
            print('Loading model from: ', load_path)
            peft_config = PeftConfig.from_pretrained(load_path, is_trainable=True)
            model = RobertaForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
            self.model = PeftModel.from_pretrained(model, load_path, is_trainable=True)
            self.model.print_trainable_parameters()
            print('Model reloaded successfully')
            print('=====================')

   

        


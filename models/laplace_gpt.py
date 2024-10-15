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
    PeftConfig,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaForCausalLM, LlamaTokenizer
)

class CustomLMHead_lora(torch.nn.Module):
    def __init__(self, original_lm_head, id_list, accelerator, args):
        super().__init__()
        self.id_list = id_list
        
        # Trim the lm_head linear weights
        original_weight = original_lm_head.weight[id_list, :].clone()
        self.linear = torch.nn.Linear(in_features=original_weight.shape[1], out_features=len(id_list), bias=False).to(accelerator.device)
        self.linear.weight.data = original_weight.to(torch.float32)
        self.linear.weight.requires_grad = False

        self.lora_dropout = original_lm_head.lora_dropout['default']

        original_lora_A_weight = original_lm_head.lora_A["default"].weight.clone()
        self.lora_A = torch.nn.Linear(in_features=original_lora_A_weight.shape[1], out_features=original_lora_A_weight.shape[0], bias=False).to(accelerator.device)
        self.lora_A.weight.data = original_lora_A_weight.to(torch.float32)
        # if args.laplace_sub == 'all':
        self.lora_A.weight.requires_grad = True
        
        # Trim the lora_B weights
        original_lora_B_weight = original_lm_head.lora_B["default"].weight[id_list, :].clone()
        self.lora_B = torch.nn.Linear(in_features=original_lora_B_weight.shape[1], out_features=len(id_list), bias=False).to(accelerator.device)
        self.lora_B.weight.data = original_lora_B_weight.to(torch.float32)
        self.lora_B.weight.requires_grad = True
        
        self.scaling = args.lora_alpha / args.lora_r

    def forward(self, x):
        linear_out = self.linear(x[:, -1, :].to(torch.float32))
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x[:, -1, :].to(torch.float32))))
        return linear_out + lora_out * self.scaling


class WrappedModel(torch.nn.Module):
    def __init__(self, model, tokenizer, accelerator, args):
        super().__init__()

        if args.dataset == 'boolq':
            self.id_list = [tokenizer.encode('False')[1], tokenizer.encode('True')[1]]
        elif args.dataset == 'openbookqa':
            self.id_list = [tokenizer.encode('A')[1], tokenizer.encode('B')[1], tokenizer.encode('C')[1], tokenizer.encode('D')[1]]
        elif 'ARC' in args.dataset:
            self.id_list = [tokenizer.encode('A')[1], tokenizer.encode('B')[1], tokenizer.encode('C')[1], tokenizer.encode('D')[1]]
        elif 'winogrande' in args.dataset:
            self.id_list = [tokenizer.encode('A')[1], tokenizer.encode('B')[1]]

        if args.lm_head:
            original_lm_head = model.base_model.model.lm_head
            model.base_model.model.lm_head = CustomLMHead_lora(original_lm_head, self.id_list, accelerator, args).to(accelerator.device) 
        
        self.model = model
        self.args = args

        # model.print_trainable_parameters()
        # print(self.model)

    def forward(self, **kwargs):
        kwargs.pop('labels', None)
        output_dict = self.model(**kwargs)
        logits = output_dict['logits']
        if self.args.lm_head:
            selected_logits = logits
        else:
            selected_logits = logits[:, -1, self.id_list]
        return selected_logits.to(torch.float32)
    
class WrappedModel_normal(torch.nn.Module):
    def __init__(self, model, tokenizer, accelerator, args):
        super().__init__()
        if args.dataset == 'boolq':
            self.id_list = [tokenizer.encode('False')[1], tokenizer.encode('True')[1]]
        elif args.dataset == 'openbookqa':
            self.id_list = [tokenizer.encode('A')[1], tokenizer.encode('B')[1], tokenizer.encode('C')[1], tokenizer.encode('D')[1]]
        elif 'ARC' in args.dataset:
            self.id_list = [tokenizer.encode('A')[1], tokenizer.encode('B')[1], tokenizer.encode('C')[1], tokenizer.encode('D')[1]]
        elif 'winogrande' in args.dataset:
            self.id_list = [tokenizer.encode('A')[1], tokenizer.encode('B')[1]]
        self.model = model

    def forward(self, **kwargs):
        kwargs.pop('labels', None)
        output_dict = self.model(**kwargs)
        logits = output_dict['logits']
        selected_logits = logits[:, -1, self.id_list]
        output_dict['logits'] = selected_logits
        return output_dict['logits'].to(torch.float32)

class laplace_gpt(nn.Module):
    def __init__(self, args, accelerator=None, tokenizer=None, **kwargs) -> None:
        super().__init__()
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if args.load_model_path is not None:
            # if accelerator.is_main_process:
            # # Load the pretrained checkpoint first
            load_path = f'checkpoints/{args.dataset}/{args.model}/{args.model}/{args.load_model_path}'
            peft_config = PeftConfig.from_pretrained(load_path, is_trainable=True)
            # peft_config.inference_mode = False
            model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, load_in_8bit=args.load_in_8bit)
            # model = prepare_model_for_kbit_training(model)
            self.model = PeftModel.from_pretrained(model, load_path, is_trainable=True)
            self.model.print_trainable_parameters()
            # print('======')
            # if 'last_layer' in args.laplace_sub:
            #     assert args.laplace_prior == 'homo'
            # for name, param in self.model.named_parameters():
            #     param.requires_grad = False
            #     if 'lora' in name:
            #         if 'all' in args.laplace_sub:
            #             param.requires_grad = True
            # self.model.print_trainable_parameters()
            # self.model = WrappedModel(self.model, tokenizer, accelerator, args)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model, load_in_8bit=True)
            target_modules=['v_proj','q_proj']
            if args.lm_head:
                target_modules.append('lm_head')
            peft_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=target_modules)
            self.model = get_peft_model(model, peft_config)            
            self.model.print_trainable_parameters()
            self.model = WrappedModel_normal(self.model, tokenizer, accelerator, args)

    def forward(self, batch, returnt='logits'):
        logits = self.model(**batch)

        if returnt == 'logits':
            return logits
        else:
            return NotImplementedError("Unsupported return type")
        
    def reload(self, args, accelerator):
        super().__init__()
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            load_path = f'checkpoints/{args.dataset}/{args.backbone}/{args.model}/{args.load_model_path}'
            peft_config = PeftConfig.from_pretrained(load_path, is_trainable=True)
            # peft_config.inference_mode = False
            model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, load_in_8bit=args.load_in_8bit)
            # model = prepare_model_for_kbit_training(model)
            self.model = PeftModel.from_pretrained(model, load_path, is_trainable=True)
            self.model.print_trainable_parameters()



   

        


import torch.nn as nn

from run import get_modelwrapper

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    PeftModel,
    PeftConfig,
)


class CausalLM(nn.Module):
    def __init__(self, args, accelerator=None, **kwargs) -> None:
        super().__init__()
        if accelerator is not None:
            accelerator.wait_for_everyone()

        bnb_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit)
        if args.load_model_path is not None:
            model = AutoModelForCausalLM.from_pretrained(
                args.load_model_path, quantization_config=bnb_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, quantization_config=bnb_config
            )
        if args.apply_classhead_lora:
            target_modules = ["q_proj", "v_proj", "lm_head"]
        elif args.apply_qkv_head_lora:
            target_modules = ["q_proj", "v_proj", "k_proj", "lm_head"]
        else:
            target_modules = ["q_proj", "v_proj"]

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        self.model = get_peft_model(model, peft_config)
        self.peft_config = peft_config

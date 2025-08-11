import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
from dataclasses import dataclass, field
from .wrapperbase import WrapperBase
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from run.evaluation import *

from peft.tuners.lora import LoraLayer, Linear
from peft.tuners.lora.bnb import Linear8bitLt

from transformers import PreTrainedModel
from peft.config import PeftConfig

from .blob import blob_linear_forward, sample, blob_8bitlinear_forward, BLoBConfig

import logging
import time


## Model Specific Argument Parsing
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='standard training, MLE')
    add_management_args(parser)
    add_experiment_args(parser)

    parser.add_argument('--bayes-train-n-samples', type=int, default=1)
    parser.add_argument('--bayes-eval-n-samples', type=int, default=1,
                        help="Number of samples to use during evaluation when training.")
    parser.add_argument('--bayes-eval-n-samples-final', type=int, default=10,
                        help="Number of samples to use during evaluation.")
    
    parser.add_argument('--bayes-eps', type=float, default=0.05)         
    parser.add_argument('--bayes-gamma', type=float, default=8)
    parser.add_argument('--bayes-kllr', type=float, default=0.02)
    parser.add_argument('--bayes-beta', type=float, default=0.2)
    parser.add_argument('--bayes-final-beta', type=float, default=0.18)
    parser.add_argument('--bayes-flipout', type=bool, default=True)
    parser.add_argument('--bayes-inference-notsample', action='store_true',
                        help='Whether to sample during inference.')
    parser.add_argument('--bayes-klreweighting', action='store_true',
                        help='Whether to use reweighting.')

    parser.add_argument('--th', type=float, default=0.01)   
    parser.add_argument('--iter', type=int, default=10)   
    return parser


class TFBLoRAACC(WrapperBase):
    """MLE with Static Variance Inference model."""
    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, args, accelerator, adapter_name: str = "default"):
        super().__init__(model, peft_config, args, accelerator, adapter_name)

        # same BLoBConfig as in the BLoB model, 
        # only used during inference.
        self.blobconfig = BLoBConfig(bayes_eps = self.args.bayes_eps, bayes_beta = self.args.bayes_beta)
        self.load_lora_path = self.args.load_lora_path
        self.train_n_samples = self.args.bayes_train_n_samples
        
        if self.args.load_lora_huggingface_repo is not None:
            repo_id = self.args.load_lora_huggingface_repo
            subfolder = self.load_lora_path 
            self.load_adapter(repo_id, adapter_name='default', subfolder=subfolder)
            print(f'LoRA Model loaded successfully from HF repo: {repo_id}/{subfolder}')
            print('=====================')

        elif self.load_lora_path is not None:
            self.load_adapter(self.load_lora_path, adapter_name='default')
            print(f'LoRA Model loaded successfully from local path: {self.load_lora_path}')
            print('=====================')
    
        self.lora_layers = []

        def extract_lora_layers(module):
            for child in module.children():
                if isinstance(child, (Linear8bitLt, Linear)):
                    self.lora_layers.append(child)
                else:
                    extract_lora_layers(child)
        extract_lora_layers(self)

        self._modify_lora_layers(self.lora_layers)
                
    def fit(self, anchor_loader, target_ratio=0.01, max_iters=10):
    
        # Initialize the binary search range [low, high]
        low, high = 0.001, self.args.bayes_beta
        best_bayes_beta = high  # Initialize the best value as the upper bound
        
        # Perform binary search
        for _ in range(max_iters):
            mid = (low + high) / 2
            self.args.bayes_beta = mid
            self._update_lora_layers(self.lora_layers, self.args.bayes_beta)
            print(f"Current bayes_beta: {self.args.bayes_beta}")
            
            all_predicted_classes = []
            all_logits = []
            
            # Evaluate the current bayes_beta
            with torch.no_grad() and torch.inference_mode():
                for i, batch in enumerate(anchor_loader): 
                    logits = self.forward_logits(batch, sample=True, n_samples=self.train_n_samples)
                    output = torch.softmax(logits, dim=-1).mean(1)
                    predicted_classes = torch.argmax(output, dim=-1)
                    
                    all_predicted_classes.append(predicted_classes)
                    all_logits.append(logits)
                    
            all_predicted_classes = torch.cat(all_predicted_classes, dim=0)
            all_logits = torch.cat(all_logits, dim=0)
            
            # Calculate the changed class ratio
            changed_class_ratio = (all_predicted_classes != self.all_ori_predicted_classes).float().mean().item()
            print(f"Changed_class ratio: {changed_class_ratio * 100}%")
            print('=====================')
            
            # Adjust the binary search range
            if changed_class_ratio > target_ratio:
                best_bayes_beta = mid  # Update the best value
                high = mid  # Narrow the search range to the lower half
            else:
                low = mid  # Narrow the search range to the upper half
                
            # Termination condition: stop if the range is sufficiently small
            if high - low < 5e-4:  # Precision threshold
                break
        
        # Set the final bayes_beta to the best value if we've exited the loop without finding an optimal value
        self.args.bayes_beta = best_bayes_beta
        self._update_lora_layers(self.lora_layers, self.args.bayes_beta)
        print(f"Optimal bayes_beta: {self.args.bayes_beta}")
        return 1

    
    def sample(self, module, status = True):
        """
        Set the sampling status of the model.
        """
        for name, child in module.named_children():
            if isinstance(child, LoraLayer) and isinstance(child, Linear):
                child.sample(status)
            else:
                self.sample(child, status)
                
    def fit_evaluate(self):
        starttime = time.time()
        if self.accelerator.is_local_main_process:
            save_folder = f'checkpoints/{self.args.modelwrapper}/{self.args.model}/{self.args.dataset}/{self.args.log_path}'
            create_if_not_exists(save_folder)
            logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO, filename=save_folder+'/log.txt')

        if self.args.iter == 0:
            self._update_lora_layers(self.lora_layers, self.args.bayes_beta)
        else:
            all_ori_predicted_classes = []
            all_ori_probs = []
            print(self.args.bayes_beta)

            with torch.no_grad() and torch.inference_mode():
                for i, batch in enumerate(self.anchor_loader): 
                    self._update_lora_layers(self.lora_layers, 0)
                    ori_logits = self.forward_logits(batch, sample=True, n_samples=1)
                    ori_probs = torch.softmax(ori_logits, dim=-1).mean(1)
                    ori_predicted_classes = torch.argmax(ori_probs, dim=-1)
                    
                    all_ori_predicted_classes.append(ori_predicted_classes)
                    all_ori_probs.append(ori_probs)

            self.all_ori_probs = torch.cat(all_ori_probs, dim=0)
            self.all_ori_predicted_classes = torch.cat(all_ori_predicted_classes, dim=0)
            self.fit(self.anchor_loader, target_ratio=self.args.th, max_iters=self.args.iter)

        endtime = time.time()
        if hasattr(self.args, 'bayes_eval_n_samples_final'):
            self.eval_n_samples = self.args.bayes_eval_n_samples_final

        val_acc, val_ece, val_nll, val_brier = self.evaluate(self.test_loader)
        logging.info(f'sigma: {self.args.bayes_beta}, val_acc: {val_acc}, val_ece: {val_ece}, val_nll: {val_nll}, val_brier: {val_brier}, time: {endtime-starttime}')
        if self.accelerator.is_local_main_process:
            if self.wandb_logger is not None:
                self.wandb_logger.log({
                    'final_val_acc': val_acc,
                    'final_val_ece': val_ece,
                    'final_val_nll': val_nll,
                    'final_val_brier': val_brier,
                                    })
        
    def _modify_lora_layers(self, layers):
        """
        Recursively go through the model and modify LoraLayer instances.
        """
        for layer in layers:
            if isinstance(layer, LoraLayer) and isinstance(layer, Linear):
                self._wrap_lora_layer_var_infer(layer)
                # modify existing methods
                setattr(layer, 'forward', blob_linear_forward.__get__(layer, layer.__class__))
                # add new methods
                setattr(layer, 'sample', sample.__get__(layer, layer.__class__))
            elif isinstance(layer, LoraLayer) and isinstance(layer, Linear8bitLt):
                self._wrap_lora_layer_var_infer(layer)
                # modify existing methods
                setattr(layer, 'forward', blob_8bitlinear_forward.__get__(layer, layer.__class__))
                # add new methods
                setattr(layer, 'sample', sample.__get__(layer, layer.__class__))
            else:
                print(layer)
                exit()
    
    def _wrap_lora_layer_var_infer(self, lora_layer):
        lora_layer.lora_A_rho = nn.ParameterDict({})
        lora_layer.bayes_eps = self.blobconfig.bayes_eps
        lora_layer.bayes_beta = self.blobconfig.bayes_beta
        lora_layer.blobsample = True
    
        for adapter_name in lora_layer._active_adapter:
            # ipdb.set_trace()
            dtype_A = lora_layer.lora_A[adapter_name].weight.dtype
            dtype_B = lora_layer.lora_B[adapter_name].weight.dtype
            lora_A = lora_layer.lora_A[adapter_name].weight.float()
            lora_B = lora_layer.lora_B[adapter_name].weight.float()
            
            # SVD the lora_B.weight 
            U, D, V = torch.svd(lora_B) # svd is not performed for half precision.

            # infer the std of the posterior
            lora_std = lora_layer.bayes_beta / (torch.tile(D.reshape(-1, 1), dims=(1, lora_layer.in_features)) + 1e-6)
            if lora_layer.bayes_eps < 0:
                lora_layer.lora_A_rho[adapter_name] = nn.Parameter(torch.log(torch.exp(lora_std)-1))
            else: 
                lora_layer.lora_A_rho[adapter_name] = nn.Parameter(torch.sqrt(lora_std))

            # recalculating the lora_A' and lora_B'
            lora_layer.lora_B[adapter_name].weight = nn.Parameter(U @ torch.diag(D)).to(dtype=dtype_B)
            lora_layer.lora_A[adapter_name].weight = nn.Parameter(V.T @ lora_A).to(dtype=dtype_A)
        
        return
    
    def _update_lora_layers(self, layers, beta):
        """
        Recursively go through the model and modify LoraLayer instances.
        """
        for layer in layers:
            if isinstance(layer, LoraLayer) and isinstance(layer, Linear):
                self._update_lora_layer_var_infer(layer, beta)
            elif isinstance(layer, LoraLayer) and isinstance(layer, Linear8bitLt):
                self._update_lora_layer_var_infer(layer, beta)
            else:
                print(layer)
                exit()
    
    def _update_lora_layer_var_infer(self, lora_layer, beta):
    
        for adapter_name in lora_layer._active_adapter:
            lora_B = lora_layer.lora_B[adapter_name].weight.float()
            
            # SVD the lora_B.weight 
            U, D, V = torch.svd(lora_B) # svd is not performed for half precision.

            # infer the std of the posterior
            lora_std = beta / (torch.tile(D.reshape(-1, 1), dims=(1, lora_layer.in_features)) + 1e-6)

            if lora_layer.bayes_eps < 0:
                lora_layer.lora_A_rho[adapter_name] = nn.Parameter(torch.log(torch.exp(lora_std)-1))
            else: 
                lora_layer.lora_A_rho[adapter_name] = nn.Parameter(torch.sqrt(lora_std))
        
        return
    
    def forward_logits(self, batch, sample=True, n_samples=1, **kwargs) -> torch.Tensor:
        if self.args.dataset_type == 'mcdataset':
            inputs, _, _ = batch
            if not sample:
                self.sample(self.base_model, False)
                output = self.base_model(**inputs)
                logits = output.logits[:, -1, self.target_ids]
                self.sample(self.base_model, True)
                return logits.unsqueeze(1)
            else:
                logits_list = []
                for _ in range(n_samples):
                    output = self.base_model(**inputs)
                    logits = output.logits[:, -1, self.target_ids]
                    logits_list.append(logits)
                return torch.stack(logits_list, dim = 1)
        else:
            if not sample:
                self.sample(self.base_model, False)
                res = self.base_model(**batch).logits
                self.sample(self.base_model, True)
                return res.unsqueeze(1)
            else:
                res = []
                for _ in range(n_samples):
                    res.append(self.base_model(**batch).logits)    
                return torch.stack(res, dim = 1)

    
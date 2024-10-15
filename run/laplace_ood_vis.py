import math
import pandas as pd
import sys
from argparse import Namespace
from typing import Tuple
from unittest import result
import time
import logging

import torch
from tqdm import tqdm
import json
import evaluate

from utils.status import ProgressBar

from run.evaluation import *
from run import *
from models.laplace_bayeslib import Laplace
from torchmetrics import Accuracy, CalibrationError
from run.evaluation import *
import numpy as np

from accelerate import Accelerator
try:
    import wandb
except ImportError:
    wandb = None



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
        x = x[:, -1, :].to(torch.float32)
        linear_out = self.linear(x)
        scaling = self.scaling

        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        result = linear_out + lora_out * scaling
        return result

class CustomLMHead_lora_hidden(torch.nn.Module):
    def __init__(self, original_lm_head, id_list, accelerator, args):
        super().__init__()

    def forward(self, x):
        x = x[:, -1, :].to(torch.float32)
        return x

class WrappedModel(torch.nn.Module):
    def __init__(self, model, accelerator, args, id_list):
        super().__init__()

        original_lm_head = model.model.lm_head
        model.model.lm_head = CustomLMHead_lora(original_lm_head, id_list, accelerator, args).to(accelerator.device) 
        
        self.model = model
        model.print_trainable_parameters()
    
    def forward(self, **kwargs):
        kwargs.pop('labels', None)
        output = self.model(**kwargs)
        logits = output.logits.to(torch.float32)
        return logits

class WrappedModel_hidden(torch.nn.Module):
    def __init__(self, model, accelerator, args, id_list):
        super().__init__()

        original_lm_head = model.model.lm_head
        model.model.lm_head = CustomLMHead_lora_hidden(original_lm_head, id_list, accelerator, args).to(accelerator.device) 
        
        self.model = model
        model.print_trainable_parameters()
    
    def forward(self, **kwargs):
        kwargs.pop('labels', None)
        output = self.model(**kwargs)
        logits = output.logits.to(torch.float32)
        return logits

    
def laplace_ood_vis(model, dataset, accelerator, args: Namespace, ood_ori_dataset=None):
    """
    The training process, including evaluations and loggers.
    
    Args:
        
        model: the model to be trained
        dataset: the dataset at hand
        args: the arguments of the current execution
    """
    save_folder = f'checkpoints/{args.dataset}/{args.model}/{args.model}/{args.log_path}'
    create_if_not_exists(save_folder)
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO, filename=save_folder+'/log.txt')
    if accelerator.is_local_main_process:
        if not args.nowand:
            assert wandb is not None, "Wandb not installed, please install it or run without wandb"
            if not args.wandb_name:
                wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
            else:
                wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=vars(args))
            args.wandb_url = wandb.run.get_url()
        print(file=sys.stderr)
    train_loader, test_loader, ood_ori_test_loader = ood_ori_dataset.train_dataloader, dataset.test_dataloader, ood_ori_dataset.test_dataloader

    if args.dataset_type == 'mcdataset':
        model.tokenizer = ood_ori_dataset.tokenizer
    step = 0

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps == 0:
        args.max_train_steps = args.n_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    args.n_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if accelerator.is_local_main_process:
        print('len(train_loader):', len(train_loader))
        print('num of epochs:', args.n_epochs)

    if 'all' not in args.laplace_sub:
        for layer in model.net.model.model.model.layers:
            for name, param in layer.named_parameters():
                param.requires_grad = False
    model.net.model.print_trainable_parameters()
    model.net.model = WrappedModel(model.net.model, accelerator, args, dataset.target_ids.squeeze(-1))

    model.net.model, train_loader, test_loader = accelerator.prepare(
        model.net.model, train_loader, test_loader
    )

    if args.testing_set == 'train_val':
        val_loader = accelerator.prepare(val_loader)

    model.net.model.eval()

    la = Laplace(model.net.model, 'classification', prior_precision=1.,
                    subset_of_weights='all',
                    hessian_structure=args.laplace_hessian)
    
    start_time = time.time()

    print('----fitting Laplace-----')
    la.fit(train_loader)

    if args.testing_set != 'val':
        raise NotImplementedError("Unsupported val set")
        # prior_precision = la.optimize_prior_precision(method='CV', val_loader=val_dataloader, link_approx='mc', log_prior_prec_min=-3, log_prior_prec_max=4, grid_size=50, n_steps=args.laplace_optim_step, lr=1e-1)
    else:
        prior_precision = la.optimize_prior_precision(method='marglik', n_steps=args.laplace_optim_step, lr=1e-1)
        print(f'prior precision: {prior_precision}')

    id_prob_list = np.array([])
    label_list = np.array([])
    mode_list = np.array([])
    samples_seen = 0
    f_mu_list = []
    f_var_list = []

    backbone = get_backbone(args=args, accelerator=accelerator, tokenizer=dataset.tokenizer)
    loss = get_loss(loss_name=args.loss)
    model = get_model(args, backbone, loss, accelerator)

    model.net.model = WrappedModel_hidden(model.net.model, accelerator, args, dataset.target_ids.squeeze(-1))
    model.net.model = accelerator.prepare(
        model.net.model
    )
    setattr(model.net.model, 'output_size', 4096)
    la.model = model.net.model


    for step, batch in tqdm(enumerate(ood_ori_test_loader)):
        # batch = {k: v.to(model.device) for k, v in batch.items()}
        if args.dataset_type == 'mcdataset':
            prompts, classes, _ = batch
            prompts = prompts.to(accelerator.device)
            labels_ = classes.to(accelerator.device)
            batch = prompts
        else:
            labels_ = batch["labels"]
            batch.pop('labels')
        with torch.no_grad():
            f_mu, f_var = la._glm_predictive_distribution(batch)
            f_mu_list.append(f_mu)
            f_var_list.append(f_var)
        
        samples = 100
        
        f_mu = f_mu.expand(samples, -1, -1)
        f_var = f_var.expand(samples, -1, -1, -1)
        logits = (f_mu + (torch.linalg.cholesky(f_var).to(f_mu.dtype) @ torch.randn_like(f_mu).unsqueeze(-1).to(f_mu.dtype).to(accelerator.device)).squeeze(-1))
        mode = f_mu
        logits, labels, mode = accelerator.gather((logits, labels_, mode))
        
        if accelerator.num_processes > 1:
            if step == len(ood_ori_test_loader) - 1:
                logits = logits[: len(ood_ori_test_loader.dataset) - samples_seen]
                labels = labels[: len(ood_ori_test_loader.dataset) - samples_seen]
                mode = mode[: len(ood_ori_test_loader.dataset) - samples_seen]
            else:
                samples_seen += labels.shape[0]

        print(logits.size())
        print(mode.size())

        id_prob_list = np.append(id_prob_list, logits.detach().cpu().numpy())
        label_list = np.append(label_list, labels.detach().cpu().numpy())
        mode_list = np.append(mode_list, mode.detach().cpu().numpy())

    create_if_not_exists('log-ood-detection')
    with open(os.path.join('log-ood-detection', f'{args.model}-{args.ood_ori_dataset}-{args.dataset}-{args.ood_detection_method}-representation-ID-seed{args.seed}.pkl'), 'wb') as f:
        to_dump = {"labels": labels, "representations": id_prob_list, "modes": mode_list}
        pickle.dump(to_dump, f)

    # backbone = get_backbone(args=args, accelerator=accelerator, tokenizer=dataset.tokenizer)
    # loss = get_loss(loss_name=args.loss)
    # model = get_model(args, backbone, loss, accelerator)


    # model.net.model = WrappedModel_hidden(model.net.model, accelerator, args, dataset.target_ids.squeeze(-1))
    # model.net.model = accelerator.prepare(
    #     model.net.model
    # )
    # la.model = model.net.model
    metric_kwargs = {"task": "multiclass", "num_classes": args.ood_ori_outdim}
    samples_seen = 0
    f_mu_list = []
    f_var_list = []
    id_prob_list = np.array([])
    label_list = np.array([])
    mode_list = np.array([])

    for step, batch in tqdm(enumerate(test_loader)):
        # batch = {k: v.to(model.device) for k, v in batch.items()}
        if args.dataset_type == 'mcdataset':
            prompts, classes, _ = batch
            prompts = prompts.to(accelerator.device)
            labels_ = classes.to(accelerator.device)
            batch = prompts
        else:
            labels_ = batch["labels"]
            batch.pop('labels')
        with torch.no_grad():
            f_mu, f_var = la._glm_predictive_distribution(batch)
            f_mu_list.append(f_mu)
            f_var_list.append(f_var)
        
        samples = 100
        
        f_mu = f_mu.expand(samples, -1, -1)
        f_var = f_var.expand(samples, -1, -1, -1)
        logits = f_mu + (torch.linalg.cholesky(f_var).to(f_mu.dtype) @ torch.randn_like(f_mu).unsqueeze(-1).to(f_mu.dtype).to(accelerator.device)).squeeze(-1)
        mode = f_mu

        logits, labels, mode = accelerator.gather((logits, labels_, mode))
        if accelerator.num_processes > 1:
            if step == len(test_loader) - 1:
                logits = logits[: len(test_loader.dataset) - samples_seen]
                labels = labels[: len(test_loader.dataset) - samples_seen]
                mode = mode[: len(ood_ori_test_loader.dataset) - samples_seen]
            else:
                samples_seen += labels.shape[0]
        id_prob_list = np.append(id_prob_list, logits.detach().cpu().numpy())
        label_list = np.append(label_list, labels.detach().cpu().numpy())
        mode_list = np.append(mode_list, mode.detach().cpu().numpy())
            


    # log the scores
    create_if_not_exists('log-ood-detection')
    with open(os.path.join('log-ood-detection', f'{args.model}-{args.ood_ori_dataset}-{args.dataset}-{args.ood_detection_method}-representation-OOD-seed{args.seed}.pkl'), 'wb') as f:
        to_dump = {"labels": labels, "representations": id_prob_list, "modes": mode_list}
        pickle.dump(to_dump, f)

    




import math
import pandas as pd
import sys
from argparse import Namespace
from typing import Tuple
from unittest import result
import time

import logging
from tqdm import tqdm

from utils.status import ProgressBar

from run.evaluation import *

from accelerate import Accelerator

try:
    import wandb
except ImportError:
    wandb = None

def ood_eval(model, dataset, accelerator, args: Namespace, ood_ori_dataset):
    """
    The training process, including evaluations and loggers.
    
    Args:
        model: the model to be trained
        dataset: the dataset at hand
        args: the arguments of the current execution
    """

    if accelerator.is_local_main_process:
        print(args)
        save_folder = f'checkpoints/{args.dataset}/{args.model}/{args.model}/{args.log_path}'
        create_if_not_exists(save_folder)
        logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO, filename=save_folder+'/log.txt')
        if not args.nowand:
            assert wandb is not None, "Wandb not installed, please install it or run without wandb"
            if not args.wandb_name:
                wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
            else:
                wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=vars(args))
            args.wandb_url = wandb.run.get_url()
        print(file=sys.stderr)

    test_loader = dataset.test_dataloader
    ood_ori_test_loader = ood_ori_dataset.test_dataloader

    model.tokenizer = dataset.tokenizer
    model.net.target_ids = dataset.target_ids.squeeze(-1)
    model.net.model.target_ids = dataset.target_ids.squeeze(-1)


    model.net, test_loader, ood_ori_test_loader = accelerator.prepare(
        model.net,  test_loader, ood_ori_test_loader
    )
    start_time = time.time()

    model.net.module.eval()
    if args.model.startswith('blob'):
        start_time = time.time()
        model.eval_n_samples = args.eval_n_samples_final
        test_acc, test_ece, test_nll, test_brier, ood_acc, ood_auc = evaluate_ood_detection(model, dataset, ood_ori_dataset, test_loader, ood_ori_test_loader, accelerator, args)
        if accelerator.is_local_main_process:
            wandb.log({'test_acc': test_acc, 'test_ece': test_ece, 'test_nll': test_nll, 'test_brier':test_brier, "ood_acc": ood_acc, "ood_auc": ood_auc})
            logging.info(f'test_acc: {test_acc}, test_ece: {test_ece}, test_nll: {test_nll}, test_brier: {test_brier}, ood_acc: {ood_acc}, ood_auc: {ood_auc}')
            end_time = time.time()
            time_seconds = end_time - start_time
            time_minutes = time_seconds / 60
            print(time_minutes)
        
    else:
        test_acc, test_ece, test_nll, test_brier, ood_acc, ood_auc = evaluate_ood_detection(model, dataset, ood_ori_dataset, test_loader, ood_ori_test_loader, accelerator, args)
        if accelerator.is_local_main_process:
            wandb.log({'test_acc': test_acc, 'test_ece': test_ece, 'test_nll': test_nll, 'test_brier':test_brier, "ood_acc": ood_acc, "ood_auc": ood_auc})
            logging.info(f'test_acc: {test_acc}, test_ece: {test_ece}, test_nll: {test_nll}, test_brier: {test_brier}, ood_acc: {ood_acc}, ood_auc: {ood_auc}')

    # OOD should be done in the same way as the in-distribution dataset using checkpoint trained by the in-distribution dataset.

    # checkpointing the backbone model.
    if args.checkpoint: # by default the checkpoints folder is checkpoints
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_folder = f'checkpoints/{args.dataset}/{args.model}/{args.model}/{args.checkpoint_dic_name}'
            create_if_not_exists(save_folder)
            accelerator.unwrap_model(model.net).model.save_pretrained(save_folder, save_function=accelerator.save)

    if not args.nowand:
        if accelerator.is_local_main_process:
            wandb.finish()


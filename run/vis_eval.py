import numpy  # needed (don't change it)
import importlib
import os
import socket
import sys
from ipdb import iex
import copy

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
sys.path.append(project_path + '/datasets')
sys.path.append(project_path + '/backbones')
sys.path.append(project_path + '/models')
sys.path.append(project_path + '/main')

import datetime
import uuid
from argparse import ArgumentParser

import setproctitle
import torch
from utils.args import add_management_args, add_experiment_args
# from utils.continual_training import train as ctrain
from run.ood_eval import ood_eval
from run.laplace_train import laplace_train_old
from run.laplace_ood_eval import laplace_ood_eval
from run.laplace_ood_vis import laplace_ood_vis
from run import get_dataset, get_model

from accelerate.utils import set_seed
from accelerate import Accelerator

import sys
from argparse import Namespace
import time

import logging

from utils.status import ProgressBar

from run.evaluation import *

from accelerate import Accelerator

try:
    import wandb
except ImportError:
    wandb = None

import ipdb

@iex
def vis_eval(model, dataset, accelerator, args: Namespace):
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
        print(file=sys.stderr)

    test_loader = dataset.test_dataloader

    model.tokenizer = dataset.tokenizer
    model.net.target_ids = dataset.target_ids.squeeze(-1)
    model.net.model.target_ids = dataset.target_ids.squeeze(-1)

    # forward hook to get the embeddings
    # now ONLY supports the setting of single GPU.
    embeddings = {}
    def get_activation(name):
        def hook(model, input, output):
            ######### Sample Code of Transformer Block in Llamma2:
            ## h = x + self.attention(
            ## self.attention_norm(x), start_pos, freqs_cis, mask
            ## )
            ## out = h + self.feed_forward(self.ffn_norm(h))

            if name not in embeddings:
                embeddings[name] = [output[0][:, -1, :].detach().cpu().numpy()]
            else:
                embeddings[name].append(output[0][:, -1, :].detach().cpu().numpy())
        return hook
    
    # register the hook for the embeddings.
    model.net.model.base_model.model.model.layers[31].register_forward_hook(get_activation('31'))    

    model.net, test_loader = accelerator.prepare(model.net,  test_loader)

    model.net.module.eval()
    if args.model.startswith('blob'):
        model.eval_n_samples = args.eval_n_samples_final

    samples_seen = 0
    all_labels, all_logits_samples, all_logits_mode = [], [], []

    # For Collecting the embeddings and logits (samples).
    for step, batch in enumerate(test_loader):
        # # for debugging.
        # if step > 4:
        #     break
        with torch.no_grad() and torch.inference_mode():
            if args.dataset_type == 'mcdataset':
                _, labels, _ = batch
                logits_samples = model(batch, sample=True).detach()
            else:
                logits_samples = model(batch, sample=True).detach()
                labels = batch["labels"]
            # logits_samples, labels = accelerator.gather([logits_samples, labels])
            if accelerator.num_processes > 1:
                if step == len(test_loader) - 1:
                    labels = labels[: len(test_loader.dataset) - samples_seen]
                    logits_samples = logits_samples[: len(test_loader.dataset) - samples_seen]
                else:
                    samples_seen += labels.shape[0]
            
            all_labels.append(labels)
            all_logits_samples.append(logits_samples)
    
    # clear the embedding cache
    embeddings_samples = embeddings
    embeddings = {}
    
    # For Collecting the embeddings and logits (Mode)
    for step, batch in enumerate(test_loader):
        # # for debugging.
        # if step > 4:
        #     break
        with torch.no_grad() and torch.inference_mode():
            if args.dataset_type == 'mcdataset':
                _, labels, _ = batch
                logits_mode = model(batch, sample=False).detach()
            else:
                logits_mode = model(batch, sample=False).detach()
            # logits_mode = accelerator.gather([logits_mode])
            if accelerator.num_processes > 1:
                if step == len(test_loader) - 1:
                    logits_mode = logits_mode[: len(test_loader.dataset) - samples_seen]
                else:
                    samples_seen += labels.shape[0]

            all_logits_mode.append(logits_mode)
    
    # collect all the results.
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    all_logits_mode = torch.cat(all_logits_mode, dim=0).cpu().numpy()
    all_logits_samples = torch.cat(all_logits_samples, dim=0).cpu().numpy()
    all_embeddings_mode = np.vstack(embeddings['31'])
    # all_embeddings_samples = np.stack(embeddings_samples['31'], axis=0)
        
    chunk_size = model.eval_n_samples
    chunked_embedding_sampled = [np.stack(embeddings_samples['31'][i:i + chunk_size], axis=1) for i in range(0, len(embeddings_samples['31']), chunk_size)]
    all_embeddings_samples = np.concatenate(chunked_embedding_sampled, axis=0)

    all_dic = {
        'labels': all_labels, 
        'logits_mode': all_logits_mode, 
        'logits_samples': all_logits_samples,
        'embeddings_samples': all_embeddings_samples,
        'embeddings_mode': all_embeddings_mode
    }
    
    if accelerator.is_local_main_process:
        with open(f'all_dic-{args.model}-{args.dataset}.pkl', 'wb') as f:
            pickle.dump(all_dic, f)

    ## debugging code.
    # if accelerator.is_local_main_process:
    #     ipdb.set_trace()


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description='Bayesian-Visualization', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.')
    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]

    # add model-specific arguments
    mod = importlib.import_module('models.' + args.model)
    get_parser = getattr(mod, 'get_parser')
    parser = get_parser() # the real parsing happens. 
    args = parser.parse_args()

    # set random seed
    if args.seed is not None:
        set_seed(args.seed)

    return args

# @iex
def main(args=None):
    lecun_fix()
    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")
    

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    if args.atk:
        accelerator = None
    else:
        accelerator = Accelerator()

    dataset = get_dataset(args.dataset_type, accelerator, args)
    dataset.get_loaders()
    args.outdim = dataset.num_labels 
    args.num_samples = dataset.num_samples
    backbone = get_backbone(args=args, accelerator=accelerator, tokenizer=dataset.tokenizer)

    loss = get_loss(loss_name=args.loss)
    model = get_model(args, backbone, loss, accelerator)
    
    # set job name
    setproctitle.setproctitle('{}_{}_BLoB-lora-Vis'.format(args.model, args.dataset))

    # visualize the model's output
    vis_eval(model, dataset, accelerator, args)
    
    # if args.laplace_vis:
    #     laplace_ood_vis(model, dataset, accelerator, args, ood_ori_dataset)
    # if args.ood_ori_dataset is not None and args.laplace_train:
    #     laplace_ood_eval(model, dataset, accelerator, args, ood_ori_dataset)
    # elif args.laplace_train:
    #     laplace_train_old(model, dataset, accelerator, args)
    # # elif args.atk:
    # #     atk(model, dataset, accelerator, args)
    # elif args.ood_ori_dataset is not None:
    #     ood_eval(model, dataset, accelerator, args, ood_ori_dataset)
    # else:
    #     train(model, dataset, accelerator, args)


if __name__ == '__main__':
    main()

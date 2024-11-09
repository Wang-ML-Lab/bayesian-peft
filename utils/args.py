# Argments for the experiments, including the model, dataset, optimizer, etc.
# The organization of this file is inspired by the "mammonth", codebase of DER++ (https://github.com/aimagelab/mammoth)

from argparse import ArgumentParser
import math

def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.

    Args:
        parser: the parser instance
    """
    # Two datasets need to be explicitly specified.
    parser.add_argument('--dataset', type=str, 
                        help='Which dataset to perform training on.')
    parser.add_argument('--dataset-type', type=str, required=True,
                        help='Which dataset to perform training on.')
    parser.add_argument('--max-seq-len', type=int, default=512)
    parser.add_argument('--modelwrapper', type=str, required=True,
                        help='Model name, one of the following: MLE, MAP, Deep Ensemble, Batch Ensemble, BLoB LoRA')
    parser.add_argument('--model', type=str, required=True,
                        help='Backbone type, one of the following: roberta-base, roberta-large')
    parser.add_argument('--model-type', type=str, required=True,
                        help='Backbone type, one of the following: roberta-base, roberta-large')
    parser.add_argument('--load-in-8bit', type=bool, default=True, 
                        help='Whether to load the model in 8-bit.')
    
    # Optimization-specfiic arguments
    parser.add_argument('--loss', type=str, default='nll',
                        help='Loss name')
    parser.add_argument('--n-epochs', type=int, default=0,
                        help='number of epochs.')
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Total number of training steps to perform. If provided, overrides n-epochs.",
    )
    parser.add_argument(
        "--eval-per-steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--early-stop-steps",
        type=int,
        default=0,
    )
    parser.add_argument('--batch-size', type=int,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--opt', type=str, default='adamw',
                        help='Optimizer type.')
    parser.add_argument('--opt-wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--warmup-ratio', type=float, default=0,
                        help='warmup ratio.')
    parser.add_argument('--adam-epsilon', type=float, default=1e-06, 
                        help='default adam epsilon.')
    parser.add_argument('--use-slow-tokenizer', action='store_true', 
                        help='Use slow tokenizer.')
    parser.add_argument('--add-space', action='store_true', 
                        help='Add space between the prompt and the input.')
    parser.add_argument('--is_s2s', action='store_true', 
                        help='Whether the model is a sequence-to-sequence model.')
    parser.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument('--eval-steps', type=int, default=0,
                        help='set 0 to disable')
    parser.add_argument('--num-bins', type=int, default=15,
                        help='num of bins in ECE computation.')
    parser.add_argument('--load-model-path', type=str, default=None)
    parser.add_argument('--load-lora-path', type=str, default=None)
    parser.add_argument('--log-path', type=str, default='default')
    parser.add_argument('--lm-head', action='store_true')
    parser.add_argument("--testing_set", type=str, default='val')
    parser.add_argument("--ood-ori-dataset", type=str, default=None)
    
    # LoRA arguments
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--lora-dropout', type=float, default=0)
    parser.add_argument('--apply-classhead-lora', action='store_true',
                        help='Whether to apply lora on the classhead of model.')
    parser.add_argument('--apply-qkv-head-lora', action='store_true',
                        help= 'Whether to apply lora on the qkv and lm_head of model.')


def add_management_args(parser: ArgumentParser) -> None:
    """
    Arguments for the management of the experiments, e.g., seed, logging, wandb, etc.
    """
    parser.add_argument('--seed', type=int, default=1234,
                        help='The random seed.')
    parser.add_argument('--evaluate', action='store_true',
                        help='Whether to evaluate the model during training.')
    parser.add_argument('--evaluate-uncertainty', action='store_true',
                        help='Whether to evaluate the model uncertainty during training.')
    parser.add_argument('--evaluate-uncertainty-reduction', type=str, default='mean',
                        help='Whether evaluate the mean of the uncertainty.')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set for each epoch.')
    parser.add_argument('--validation-perc', type=float, default=0.1,
                        help='percentage of the validation data.')
    parser.add_argument('--checkpoint', action='store_true',
                        help='Whether checkpoint the model backbone parameters.')
    parser.add_argument('--checkpoint-name', type=str, default='default',
                        help= 'The name of the dictionary to save the checkpoint.')
    
    # Arguments Weght & Bias logging tool.
    parser.add_argument('--nowand', action='store_true', help='Inhibit wandb logging')
    parser.add_argument('--wandb-entity', type=str, default='<your_wandb_account>', help='Wandb entity')
    parser.add_argument('--wandb-project', type=str, default='Bayes LoRA', help='Wandb project name')
    parser.add_argument('--wandb-name', type=str, default='', help="Wandb run's name")


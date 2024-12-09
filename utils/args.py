from argparse import ArgumentParser
import math


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.

    Args:
        parser: the parser instance
    """
    # Dataset-related arguments
    parser.add_argument(
        "--dataset", 
        type=str, 
        help="The name of the dataset to use for training (e.g., IMDb, SST-2)."
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        required=True,
        help="The type of dataset to use. Options are 'mcdataset' (Multi-choices dataset) or 'bertds' (BERT-style dataset).",
    )
    parser.add_argument(
        "--max-seq-len", 
        type=int, 
        default=300, 
        help="Maximum sequence length for input tokenization. Default is 300."
    )

    # Model-related arguments
    parser.add_argument(
        "--modelwrapper",
        type=str,
        required=True,
        help="The type of model wrapper to use. Options: 'mle' (Maximum Likelihood Estimation), "
             "'map' (Maximum A Posteriori), 'deepensemble' (Deep Ensemble), 'mcdropout' (Monte Carlo Dropout), "
             "'BLoB' (Bayesian Low-rank Adaptation by Backpropagation)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the Hugging Face model to use (e.g., meta-llama/Llama-2-7b-hf, roberta-base).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="The model's backbone type. Options: 'seqcls' (e.g., RoBERTa for sequence classification), "
             "'causallm' (e.g., LLaMA, Mistral for causal language modeling).",
    )
    parser.add_argument(
        "--load-in-8bit",
        type=bool,
        default=True,
        help="Whether to load the model weights in 8-bit precision for memory optimization. Default is True.",
    )

    # Optimization-specific arguments
    parser.add_argument(
        "--loss", 
        type=str, 
        default="nll", 
        help="Loss function to use. Default is 'nll' (negative log-likelihood)."
    )
    parser.add_argument(
        "--n-epochs", 
        type=int, 
        default=0, 
        help="Number of training epochs. Set to 0 if using `--max-train-steps`."
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Total number of training steps. Overrides `--n-epochs` if set. Default is 0.",
    )
    parser.add_argument(
        "--eval-per-steps",
        type=int,
        default=500,
        help="Frequency of evaluation during training, measured in steps. Default is 500.",
    )
    parser.add_argument(
        "--early-stop-steps",
        type=int,
        default=0,
        help="Number of steps without improvement before early stopping. Default is 0 (disabled).",
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        help="Batch size for training and evaluation."
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.0001, 
        help="Learning rate for the optimizer. Default is 0.0001."
    )
    parser.add_argument(
        "--opt", 
        type=str, 
        default="adamw", 
        help="Optimizer type. Default is 'adamw'."
    )
    parser.add_argument(
        "--opt-wd", 
        type=float, 
        default=0.0, 
        help="Weight decay for the optimizer. Default is 0.0."
    )
    parser.add_argument(
        "--warmup-ratio", 
        type=float, 
        default=0, 
        help="Proportion of total training steps to use for learning rate warmup. Default is 0."
    )
    parser.add_argument(
        "--adam-epsilon", 
        type=float, 
        default=1e-06, 
        help="Epsilon value for the Adam optimizer. Default is 1e-6."
    )
    parser.add_argument(
        "--use-slow-tokenizer", 
        action="store_true", 
        help="Use a slow tokenizer (not backed by Rust). Recommended for debugging."
    )
    parser.add_argument(
        "--add-space",
        action="store_true",
        help="Insert a space between the prompt and the input sequence. Useful for specific tokenization strategies.",
    )
    parser.add_argument(
        "--is_s2s",
        action="store_true",
        help="Specify if the model is a sequence-to-sequence (encoder-decoder) model.",
    )
    parser.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help="If enabled, pad all samples to the maximum sequence length (`max_seq_len`). Otherwise, use dynamic padding.",
    )
    parser.add_argument(
        "--eval-steps", 
        type=int, 
        default=0, 
        help="Frequency of evaluation during training, measured in steps. Set to 0 to disable."
    )
    parser.add_argument(
        "--num-bins", 
        type=int, 
        default=15, 
        help="Number of bins to use for Expected Calibration Error (ECE) computation. Default is 15."
    )
    parser.add_argument(
        "--load-model-path", 
        type=str, 
        default=None, 
        help="Path to a pre-trained model checkpoint to load."
    )
    parser.add_argument(
        "--load-lora-path", 
        type=str, 
        default=None, 
        help="Path to a pre-trained LoRA checkpoint to load."
    )
    parser.add_argument(
        "--log-path", 
        type=str, 
        default="default", 
        help="Directory to save training logs. Default is 'default'."
    )
    parser.add_argument(
        "--testing_set", 
        type=str, 
        default="val", 
        help="Dataset split to use for testing. Default is 'val'."
    )
    parser.add_argument(
        "--ood-ori-dataset", 
        type=str, 
        default=None, 
        help="Optional path to an out-of-distribution dataset for evaluation."
    )

    # LoRA-specific arguments
    parser.add_argument(
        "--lora-r", 
        type=int, 
        default=8, 
        help="Rank of the LoRA decomposition. Default is 8."
    )
    parser.add_argument(
        "--lora-alpha", 
        type=int, 
        default=16, 
        help="Scaling factor for the LoRA updates. Default is 16."
    )
    parser.add_argument(
        "--lora-dropout", 
        type=float, 
        default=0, 
        help="Dropout rate for the LoRA layers. Default is 0 (no dropout)."
    )
    parser.add_argument(
        "--apply-classhead-lora",
        action="store_true",
        help="Apply LoRA on the classification head of the model.",
    )
    parser.add_argument(
        "--apply-qkv-head-lora",
        action="store_true",
        help="Apply LoRA on the query, key, value layers, and the language modeling head of the model.",
    )


def add_management_args(parser: ArgumentParser) -> None:
    """
    Arguments for the management of the experiments, e.g., seed, logging, wandb, etc.
    """
    parser.add_argument(
        "--seed", 
        type=int, 
        default=1, 
        help="Random seed for reproducibility. Default is 1."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="If enabled, evaluate the model on the validation set during training.",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="If enabled, save model checkpoints during training.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="default",
        help="Name of the checkpoint file to save. Default is 'default'.",
    )

    # WandB arguments
    parser.add_argument(
        "--nowand", 
        action="store_true", 
        help="Disable logging to Weights & Biases (WandB)."
    )
    parser.add_argument(
        "--wandb-entity", 
        type=str, 
        default="<your_wandb_account>", 
        help="WandB account name for logging experiments."
    )
    parser.add_argument(
        "--wandb-project", 
        type=str, 
        default="Bayes LoRA", 
        help="WandB project name. Default is 'Bayes LoRA'."
    )
    parser.add_argument(
        "--wandb-name", 
        type=str, 
        default="", 
        help="Name of the WandB run. Leave empty for default naming."
    )

# Bayesian PEFT
This repo contains the code for our NeurIPS 2024 paper:<br>
**BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models**<br>
Yibin Wang\*, Haizhou Shi\*, Ligong Han, Dimitris Metaxas, Hao Wang<br>
*Thirty-eighth Conference on Neural Information Processing Systems, 2024*<br>
[[Paper](https://arxiv.org/abs/2406.11675)] [OpenReview] [Talk] [Slides] [Poster]

**Important Note**: 
> This repository is currently in its initial development stage. It contains the foundational code required to replicate the in-distribution experiments (Table 1 in the paper).

> The complete version of this repository, including the implementation of LAP baseline and additional extensions, is scheduled to be uploaded by November 30, 2024.

## To Run the Code
To install the required conda environment, run:
```sh
conda create --name <env>
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets evaluate accelerate bitsandbytes jaxtyping torchmetrics setproctitle ipdb peft wandb nltk scikit-learn
```  

Before you run the code, there are a couple of settings you might want to modify: 
- `wandb_entity`: at `utils/args.py` line 139, change to your own wandb account;

The following command in the terminal could reproduce the basic results of in-distribution experiment: 
```sh
bash cmd/blob/blob-llama-all.sh
bash cmd/blob/blob-roberta-all.sh
```

> Note: The number of GPUs used for parallel training, the type of GPUs, and the model quantization settings can result in slight differences in the final performance.
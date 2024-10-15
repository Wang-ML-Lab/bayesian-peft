# Bayesian PEFT
Code for the paper [BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models](https://arxiv.org/abs/2406.11675).

> We have now provided the code for in-distribution experiments (excluding Laplace). The implementation of Laplace, along with the code for out-of-distribution experiments, is expected to be uploaded by Oct 30.

## To Run the Code
To install the required conda environment, run:
```sh
conda create --name <env>
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets evaluate accelerate bitsandbytes jaxtyping torchmetrics setproctitle ipdb peft wandb nltk scikit-learn
```  

The following command in the terminal could reproduce the basic results of in-distribution experiment: 
```sh
bash cmd/blob/blob-llama-all.sh
bash cmd/blob/blob-roberta-all.sh
```

> Note: The number of GPUs used for parallel training, the type of GPUs, and the model quantization settings can result in slight differences in the final performance.

## Code Structure
- `dataset/`: each file is an independent dataset.
    - `utils/`: utility files/functions for data manipulation.
        - `datasetbase.py`: the base class for the Dataset object
        - `dsets.py`: hacked from https://github.com/MaximeRobeyns/bayesian_lora/blob/master/examples/utils/dsets.py, support a series of sequence to sequence dataset
    - `S2ClassDataset.py`: Sequence to class dataset, for the bert-like models
    - `S2SDataset_Classification.py`: multichoice QA dataset, for the gpt-like models
- `modelwrappers/`: 
    - `wrapperbase.py`: the base class for the wrapper object, where it defines the basic requirements to be met for a new modelwrapper definition. 
- `utils/`: folders containing shared utility functions among modules or for the main procudure in `main.py`. **Before you run the code, you need create your own `Weight & Bias` account and put your user name at line 139 in the file `args.py`. **
- `main/`: folders containing training and evaluation pipelines. 
- `checkpoints/`: not shown in this repo, where by default the checkpoints and logs of the models are stored. 
- `wandb/`: not shown in this repo, where the local logs of the wandb will be stored.

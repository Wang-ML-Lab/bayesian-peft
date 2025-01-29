# Bayesian PEFT
This repo contains the code for our NeurIPS 2024 paper:<br>
**BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models**<br>
Yibin Wang\*, Haizhou Shi\*, Ligong Han, Dimitris Metaxas, Hao Wang<br>
*Thirty-eighth Conference on Neural Information Processing Systems, 2024*<br>
[[📄 Paper](https://arxiv.org/abs/2406.11675)] [[🌐 OpenReview](https://openreview.net/forum?id=MaDykgj4Ru)] [🎥 Talk] [[📑 Slides](https://nips.cc/media/neurips-2024/Slides/95507.pdf)] [[🖼️ Poster](https://nips.cc/media/PosterPDFs/NeurIPS%202024/95507.png)]

## 📖 Table of Contents
1. [⚙️ Installation](#installation)
2. [🚀 To Run the Code](#to-run-the-code)
3. [🔧 To Use the Code](#to-use-the-code)
4. [📚 References](#references)

## ⚙️ Installation
To install the required conda environment, run:
```sh
conda create --name <env>
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets evaluate accelerate bitsandbytes jaxtyping torchmetrics setproctitle peft wandb nltk scikit-learn
```  

## 🚀 To Run the Code
Before you run the code, there are a couple of settings you might want to modify: 
- `wandb_entity`: at `utils/args.py` line 139, change to your own wandb account;

We have provided the command to reproduce the results of in-distribution and out-of-distribution experiment in the `/scripts` folders. For Laplace-LoRA, we reproduced the results using the code from its official repository: [laplace-lora](https://github.com/adamxyang/laplace-lora).

### To run the in-distribution experiment, use the following script:
```sh
bash scripts/<method_name>/<method_name>-llama-all.sh
bash scripts/<method_name>/<method_name>-roberta-all.sh
```

### To run the out-of-distribution experiment, use the following script:
```sh
bash scripts/<method_name>/<method_name>-llama-ood-all.sh
```
In this script, we also demonstrate how to save and load your trained LoRA adapter. To save a LoRA checkpoint, use flag: ``--checkpoint --checkpoint-name $name``. To load a LoRA checkpoint, use flag: ``--load-lora-path checkpoints/$modelwrapper/<model_of_checkpoint>/<dataset_of_checkpoint>/<your_previous_checkpoint_name>``.

### To reproduce the results of BLoB (N=0) for in-distribution experiment, use the following script:
```sh
bash scripts/blob/blob-mean-llama-all-single-gpu.sh
```
> Note: BLoB (N=0) refers to directly using the mean of the weight distribution for prediction. To enable this mode, simply add the flag --bayes-inference-notsample to your script when running BLoB.

We also provide a script for running in-distribution experiments for BLoB on a single GPU, as an alternative to parallel training with `accelerate`. Other scripts can be adjusted accordingly for single GPU usage:
```sh
bash scripts/blob/blob-llama-all-single-gpu.sh
```

> Note: The number of GPUs used for parallel training, the type of GPUs, and the model quantization settings can result in slight differences in the final performance.



## 🔧 To Use the Code

### Overview of the WrapperBase Class
The `WrapperBase` class in `bayesian-peft/modelwrappers/wrapperbase.py` is designed as a flexible base class that integrates with various PEFT frameworks and datasets. Key features include:

* **Evaluation Metrics:** Includes accuracy, calibration error, negative log-likelihood, and Brier score.
* **Adapter Support:** Seamlessly integrates with the PEFT framework for parameter-efficient fine-tuning.
* **Optimizer and Scheduler:** Configurable optimizer and learning rate scheduler.
* **Training Loop:** Handles training and evaluation with built-in logging and metrics.

### Creating a Custom Wrapper
To implement a custom wrapper:

1. **Inherit from `WrapperBase`:** Your custom wrapper should subclass `WrapperBase`.
2. **Override `forward_logits`:** Implement how your model generates logits from input batches.
3. **Add Custom Behavior:** Extend or modify the existing methods to suit your needs.

Below is an example of creating a custom wrapper, CustomWrapper.

#### Step 1: Subclass `WrapperBase`
To create your custom wrapper, first subclass the `WrapperBase` class. This class manages training and evaluation routines, so when you create a custom wrapper, you can extend or modify any of the existing methods.

```python
from wrapperbase import WrapperBase

class CustomWrapper(WrapperBase):
    def __init__(self, model, peft_config, args, accelerator, adapter_name="default"):
        super().__init__(model, peft_config, args, accelerator, adapter_name)
        # Your custom initialization code
```

#### Step 2: Implement the `forward_logits` Method
The `forward_logits` method is used to define the forward pass for your model. It must return the logits (output) of the model, which are used to calculate the loss during training and for evaluation. Note that the `forward_logits` method is not implemented in the `WrapperBase` class; you need to implement it based on your specific requirements.
```python
def forward_logits(self, batch, sample=True, n_samples=1, **kwargs):
    # Custom logic to process the batch and return logits
    output = self.base_model(**batch)
    logits = output.logits
    return logits
```

#### Step 3: Add Custom Training and Evaluation Logic (Optional)
You can customize the training logic by overriding the `fit` method, which manages the training loop. You can modify how gradients are computed, how the model is updated, and how metrics are logged. The `evaluate` method handles the evaluation of your model. You can customize it to calculate additional metrics, apply different evaluation procedures, or modify how results are logged. You can also customize the `fit_evaluate` and `prepare_for_fit_evaluate` method to further control the procedure of training and evaluating.

For more information about the `WrapperBase` class, refer to the code provided in the project.


## 📚 References
[BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models](https://arxiv.org/abs/2406.11675)
```bib
@article{wang2024blob,
  title={BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models},
  author={Wang, Yibin and Shi, Haizhou and Han, Ligong and Metaxas, Dimitris and Wang, Hao},
  journal={arXiv preprint arXiv:2406.11675},
  year={2024}
}
```

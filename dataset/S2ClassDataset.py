from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)
from dataset.utils.datasetbase import DatasetBase
import os

class S2ClassDataset(DatasetBase):
    
    NAME = 'bertds' # used for loading the dataset
    task_info = {
        'sst2':{
            'num_labels':2,
            'tokenize_keys': ('sentence', None)
        },
        'imdb':{
            'num_labels':2,
            'tokenize_keys': ('text', None)
        },
        'ag_news':{
            'num_labels':4,
            'tokenize_keys': ('text', None)
        },
        'mnli':{
            'num_labels':3,
            'tokenize_keys': ('premise', 'hypothesis')
        },
        'yelp':{
            'num_labels':2,
            'tokenize_keys': ('text', None)
        },
        'wnli':{
            'num_labels':2,
            'tokenize_keys': ('sentence1', 'sentence2')
        },
        'mrpc':{
            'num_labels':2,
            'tokenize_keys': ('sentence1', 'sentence2')
        },
        'winogrande_s':{
            'num_labels':2,
            'tokenize_keys': ('sentence1', 'sentence2')
        },
        'rte':{
            'num_labels':2,
            'tokenize_keys': ('sentence1', 'sentence2')
        },
        'boolq':{
            'num_labels':2,
            'tokenize_keys': ('passage', 'question')
        },
        'wic':{
            'num_labels':2,
            'tokenize_keys': ('sentence1', 'sentence2')
        },
        'cola':{
            'num_labels':2,
            'tokenize_keys': ('sentence', None)
        },
        'cb':{
            'num_labels':2,
            'tokenize_keys': ("premise", "hypothesis")
        },# 250
        'ax':{
            'num_labels':2,
            'tokenize_keys': ("premise", "hypothesis")
        },# 1,459

        
    }

    def __init__(self, accelerator, args):
        super().__init__()

        self.args = args
        self.accelerator = accelerator
        # dataset_path = os.path.join(base_path_data(), 'wnli')
        # create_if_not_exists(dataset_path)
        if args.dataset in ['wnli', 'rte', 'mrpc', 'cola', 'sst2', 'qnli', 'qqp', 'mnli', 'ax']:
            self.raw_dataset = load_dataset("glue", args.dataset)
        elif args.dataset in ['cb', 'wic', 'boolq']:
            self.raw_dataset = load_dataset("super_glue", args.dataset)
        elif 'ARC' in args.dataset:
            self.raw_dataset = load_dataset('ai2_arc', args.dataset)
        elif 'winogrande' in args.dataset:
            self.raw_dataset = load_dataset('winogrande', args.dataset)
        else:
            self.raw_dataset = load_dataset(args.dataset)

        if 'ARC' in args.dataset or 'openbookqa' in args.dataset:
            self.deal_with_specical_datasets()

        # self.raw_dataset = load_dataset('glue', args.dataset)
            
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

        if accelerator is not None:
            accelerator.wait_for_everyone()

        # if accelerator.is_main_process:
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=not args.use_slow_tokenizer, trust_remote_code=True)

        
        self.num_labels = self.task_info[args.dataset]['num_labels']
        self.tokenize_keys = self.task_info[args.dataset]['tokenize_keys']
        
        self.num_samples = len(self.raw_dataset['train'])
        self.num_tests = len(self.raw_dataset['validation'])
        if accelerator is not None:
            if accelerator.is_local_main_process:
                print("=====================================")
                print(f"Loaded {args.dataset} dataset.")
                print(f"Number of samples: {self.num_samples}")
                print("=====================================")

        self.padding = "max_length" if args.pad_to_max_length else False

    def _tokenize(self, examples):
        if self.args.dataset == 'boolq' and 'llama' in self.args.model:
            texts = [f"Answer the question with only True or False: {question} Context: {passage}" for passage, question in zip(examples['passage'], examples['question'])]
            result = self.tokenizer(texts, padding=self.padding, max_length=self.args.max_length, truncation=True)
            result["labels"] = examples["label"]
        elif 'openbookqa' in self.args.dataset:
            choices_list = [' '.join(f'{label}. {text}' for label, text in zip(choices['label'], choices['text'])) for choices in examples['choices']]
            texts = [f"Select one of the choices that answers the following question: {question} Choices: {choices} Answer:" for question, choices in zip(examples['question_stem'], choices_list)]
            result = self.tokenizer(texts, padding=self.padding, max_length=self.args.max_seq_len, truncation=True)
            map_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
            result["labels"] = [map_dict[label] for label in examples["answerKey"]]
        elif 'ARC' in self.args.dataset:
            choices_list = [' '.join(f'{label}. {text}' for label, text in zip(choices['label'], choices['text'])) for choices in examples['choices']]
            texts = [f"Select one of the choices that answers the following question: {question} Choices: {choices} Answer:" for question, choices in zip(examples['question'], choices_list)]
            result = self.tokenizer(texts, padding=self.padding, max_length=self.args.max_seq_len, truncation=True)
            map_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
            result["labels"] = [map_dict[label] for label in examples["answerKey"]]
        elif 'winogrande' in  self.args.dataset:
            texts = [f"Select one of the choices that answers the following question: {question} Choices: A. {option1}. B {option2}. Answer:" for question, option1, option2 in zip(examples['sentence'], examples['option1'], examples['option2'])]
            result = self.tokenizer(texts, padding=self.padding, max_length=self.args.max_seq_len, truncation=True)
            map_dict = {"1": 0, "2": 1, "":None}
            result["labels"] = [map_dict[label] for label in examples["answer"]]
        else:
            sentence1_key, sentence2_key = self.tokenize_keys
            if sentence2_key is not None:
                result = self.tokenizer(examples[sentence1_key], examples[sentence2_key], padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_len)
                result["labels"] = examples["label"]
            else:
                result = self.tokenizer(examples[sentence1_key], padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_len)
                result["labels"] = examples["label"]
        
        return result

    def get_loaders(self):
        """
        Returns the train and test data loaders.
        """  
        if self.accelerator is not None:
            with self.accelerator.main_process_first():
                processed_datasets = self.raw_dataset.map(
                    self._tokenize,
                    batched=True,
                    remove_columns=self.raw_dataset["train"].column_names,
                    desc="Running tokenizer on dataset",
                )
        else:
            processed_datasets = self.raw_dataset.map(
                    self._tokenize,
                    batched=True,
                    remove_columns=self.raw_dataset["train"].column_names,
                    desc="Running tokenizer on dataset",
                )
        # print('====train data====')
        train_dataset = processed_datasets["train"]
        # print('====validation data====')
        processed_dataset = processed_datasets["validation_matched" if self.args.dataset == "mnli" else "validation"]

        if self.args.testing_set == 'test':
            ds = processed_dataset.train_test_split(test_size=0.5, seed=self.args.seed, shuffle=False)
            val_dataset, eval_dataset = ds["train"], ds["test"]
        elif self.args.testing_set == 'train_val':
            ds = train_dataset.train_test_split(test_size=0.2, seed=self.args.seed, shuffle=False)
            train_dataset, val_dataset = ds["train"], ds["test"]
            eval_dataset = processed_dataset
        elif self.args.testing_set == 'val':
            eval_dataset = processed_dataset

        # DataLoaders creation:
        if self.args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            if self.accelerator is not None:
                data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=(8 if self.accelerator.use_fp16 else None))
            else:
                data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=None)

        self.train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=self.args.batch_size, num_workers=self.args.num_workers
        )
        self.test_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=self.args.batch_size, num_workers=self.args.num_workers)

        if self.args.testing_set != 'val':
            self.val_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=data_collator, batch_size=self.args.batch_size, num_workers=self.args.num_workers)

    
    
    def deal_with_specical_datasets(self):
        
        # Initialize counters
        count_3_choices_train = 0
        count_5_choices_train = 0
        count_3_choices_valid = 0
        count_5_choices_valid = 0

        # Count in the training dataset
        for example in self.raw_datasets["train"]:
            if len(example['choices']['label']) == 3:
                count_3_choices_train += 1
            elif len(example['choices']['label']) == 5:
                count_5_choices_train += 1

        # Count in the validation dataset
        for example in self.raw_datasets["validation"]:
            if len(example['choices']['label']) == 3:
                count_3_choices_valid += 1
            elif len(example['choices']['label']) == 5:
                count_5_choices_valid += 1

        # Get total counts
        total_train = len(self.raw_datasets["train"])
        total_valid = len(self.raw_datasets["validation"])

        # Print counts
        print('====counts train====')
        print(f"Total number of training examples: {total_train}")
        print(f"Number of training questions with 3 choices: {count_3_choices_train}")
        print(f"Number of training questions with 5 choices: {count_5_choices_train}")

        print('====counts valid====')
        print(f"Total number of validation examples: {total_valid}")
        print(f"Number of validation questions with 3 choices: {count_3_choices_valid}")
        print(f"Number of validation questions with 5 choices: {count_5_choices_valid}")

        # Filter the examples in the training dataset
        filtered_train = self.raw_datasets["train"].filter(lambda example: len(example['choices']['label']) == 4)

        # Filter the examples in the validation dataset
        filtered_valid = self.raw_datasets["validation"].filter(lambda example: len(example['choices']['label']) == 4)

        # Filter the examples in the test dataset
        filtered_test = self.raw_datasets["test"].filter(lambda example: len(example['choices']['label']) == 4)

        # Replace the original datasets with the filtered datasets
        self.raw_datasets["train"] = filtered_train
        self.raw_datasets["validation"] = filtered_valid
        self.raw_datasets["test"] = filtered_test

        print('====counts train====')
        print(f"Total number of training examples: {len(self.raw_datasets['train'])}")
        print('====counts valid====')
        print(f"Total number of validation examples: {len(self.raw_datasets['validation'])}")

        def convert_choices_to_alpha(example):
            # Define a mapping from numerical to alphabetical labels
            mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}

            # Convert the 'label' field in 'choices'
            example['choices']['label'] = [mapping.get(label, label) for label in example['choices']['label']]

            # Convert the 'answerKey' field
            example['answerKey'] = mapping.get(example['answerKey'], example['answerKey'])

            example['choices']['text'] = [text if text.endswith('.') else text + '.' for text in example['choices']['text']]
            example['choices']['text'] = [text[0].upper() + text[1:] if text else text for text in example['choices']['text']]

            return example

        # Apply the conversion to the training, validation, and test datasets
        self.raw_datasets["train"] = self.raw_datasets["train"].map(convert_choices_to_alpha)
        self.raw_datasets["validation"] = self.raw_datasets["validation"].map(convert_choices_to_alpha)
        self.raw_datasets["test"] = self.raw_datasets["test"].map(convert_choices_to_alpha)

        print('====train data====')
        from collections import Counter

        # Initialize counters for training and validation datasets
        counter_train = Counter()
        counter_valid = Counter()

        # Count in the training dataset
        for example in self.raw_datasets["train"]:
            counter_train.update(example['answerKey'])

        # Count in the validation dataset
        for example in self.raw_datasets["validation"]:
            counter_valid.update(example['answerKey'])

        # Print the results
        print("Training dataset counts:")
        for choice, count in counter_train.items():
            print(f"Choice {choice}: {count} occurrences")

        print("Validation dataset counts:")
        for choice, count in counter_valid.items():
            print(f"Choice {choice}: {count} occurrences")

    
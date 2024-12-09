from transformers import AutoTokenizer
from dataset.utils import dsets
from dataset.utils.datasetbase import DatasetBase


class S2SDataset_Classification(DatasetBase):
    NAME = "mcdataset"  # mutil-choice dataset
    task_info = {
        "winogrande_s": {
            "num_labels": 2,
        },
        "winogrande_m": {
            "num_labels": 2,
        },
        "boolq": {
            "num_labels": 2,
        },
        "obqa": {
            "num_labels": 4,
        },
        "ARC-Easy": {
            "num_labels": 5,
        },
        "ARC-Challenge": {
            "num_labels": 5,
        },
    }

    def __init__(self, accelerator, args):
        super().__init__()

        self.args = args
        self.accelerator = accelerator

        accelerator.wait_for_everyone()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if args.dataset in ["boolq", "winogrande_m", "winogrande_s"]:
            self.tokenizer.add_eos_token = True
        self.tokenizer.pad_token = self.tokenizer.bos_token
        if args.dataset in self.task_info:
            self.num_labels = self.task_info[args.dataset]["num_labels"]
        elif args.dataset.startswith("MMLU"):
            self.num_labels = 4
        else:
            raise NotImplementedError

        if args.dataset.startswith("winogrande"):
            dset_class: dsets.ClassificationDataset = getattr(dsets, "winogrande")
            self.dset = dset_class(
                self.tokenizer,
                add_space=args.add_space,
                name=args.dataset,
                max_seq_len=args.max_seq_len,
            )
        elif args.dataset.startswith("ARC"):
            dset_class: dsets.ClassificationDataset = getattr(dsets, "arc")
            self.dset = dset_class(
                self.tokenizer,
                add_space=args.add_space,
                name=args.dataset,
                max_seq_len=args.max_seq_len,
            )
        elif args.dataset.startswith("MMLU"):
            dset_class: dsets.ClassificationDataset = getattr(dsets, "mmlu")
            self.dset = dset_class(
                self.tokenizer,
                add_space=args.add_space,
                name=args.dataset[5:],
                max_seq_len=args.max_seq_len,
            )
        else:
            dset_class: dsets.ClassificationDataset = getattr(dsets, args.dataset)
            self.dset = dset_class(
                self.tokenizer, add_space=args.add_space, max_seq_len=args.max_seq_len
            )

        if accelerator.is_local_main_process:
            print("=====================================")
            print(f"Loaded {args.dataset} dataset.")
            print("=====================================")

    def get_loaders(self):
        """
        Returns the train and test data loaders.
        """

        self.target_ids = self.dset.target_ids

        if self.args.dataset.startswith("MMLU"):
            self.train_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split="test",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            total_data_count = 0
            for batch in self.train_dataloader:
                total_data_count += batch[1].size(0)
            self.num_samples = total_data_count
            self.test_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split="test",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            return

        self.train_dataloader = self.dset.loader(
            is_s2s=self.args.is_s2s,  # sequence to sequence model?
            batch_size=self.args.batch_size,  # training batch size
            split="train",  # training split name in dset
            subset_size=-1,  # train on subset? (-1 = no subset)
        )
        total_data_count = 0
        for batch in self.train_dataloader:
            total_data_count += batch[1].size(0)
        self.num_samples = total_data_count

        self.test_dataloader = self.dset.loader(
            is_s2s=self.args.is_s2s,  # sequence to sequence model?
            batch_size=self.args.batch_size,  # training batch size
            split="validation",  # training split name in dset
            subset_size=-1,  # train on subset? (-1 = no subset)
        )

        if self.args.testing_set != "val":
            raise NotImplementedError("Only validation set is supported for now.")

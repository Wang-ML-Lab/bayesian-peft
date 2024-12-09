# Copyright (C) 2023-24 Maxime Robeyns <dev@maximerobeyns.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convenience wrappers around classification datasets
"""
import string
import re
import torch as t
import pandas as pd

from abc import abstractmethod
from enum import Enum
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset


class ClassificationDataset:
    """
    An abstract base dataset for sequence classification problems. Multiple
    choice QA problems could also be made a subclass of this class with an
    appropriate collation / formatting.
    """

    def __init__(
        self,
        dset,
        tokenizer,
        n_labels: int,
        preamble: str = "",
        add_space: bool = False,
        numerical: bool = True,
        boolean: bool = False,
        max_seq_len: int = 512,
    ):
        """
        Args:
            dset: The loaded Dataset
            tokenizer: The model tokenizer
            n_labels: The number of labels / classes for each question
            preamble: Preamble for general pre-trained / 'CausalLM' models
            add_space: Add an explicit space suffix between preamble and answer tokens.
            numerical: whether labels are numerical (0, 1, ...) or alphabetical (A, B, ...)
        """
        self.dset = dset
        self.n_labels = n_labels
        self.preamble = preamble
        self.add_space = add_space
        self.tokenizer = tokenizer
        self.numerical = numerical
        self.max_seq_len = max_seq_len

        spc = " " if self.add_space else ""
        """Token ids of class labels. Example [345, 673, 736]."""
        # TODO: return with enum for question type
        if numerical and boolean:
            raise ValueError("Question type cannot be both numerical and boolean")
        if boolean:
            labels = [f"{spc}True", f"{spc}False"]
        elif numerical:
            labels = [f"{spc}{i}" for i in range(self.n_labels)]
        else:  # alphabetical
            labels = [f"{spc}{chr(ord('A')+i)}" for i in range(self.n_labels)]
        self.target_ids = tokenizer(
            labels, return_tensors="pt", add_special_tokens=False
        ).input_ids[
            :, -1:
        ]  # assume these encode to single tokens
        """A mapping from label _indices_ to target token ids. This is only useful for CausalLM models.
        Example: {(0, 345), (1, 673), (2, 736)}
        """
        self.label2target = OrderedDict(
            [(i, self.target_ids[i]) for i in range(n_labels)]
        )
        # misnomer: should be target 2 label _index_
        self.target2label = OrderedDict(
            [(self.target_ids[i], i) for i in range(n_labels)]
        )

    @abstractmethod
    def s2s_collate_fn(self, batch):
        """Collate function for sequence to sequence models"""
        raise NotImplementedError

    def s2s_loader(self, dset: Dataset, *args, **kwargs) -> DataLoader:
        """Returns the dataloader for sequence to sequence models"""
        return t.utils.data.DataLoader(
            dset, collate_fn=self.s2s_collate_fn, *args, **kwargs
        )

    @abstractmethod
    def clm_collate_fn(self, batch):
        """Collate function for causal language models"""
        raise NotImplementedError

    def clm_loader(self, dset: Dataset, *args, **kwargs) -> DataLoader:
        """Returns the dataloader for causal language models"""
        return t.utils.data.DataLoader(
            dset, collate_fn=self.clm_collate_fn, *args, **kwargs
        )

    def loader(
        self,
        *args,
        is_s2s: bool = False,
        split: str = "train",
        subset_size: int = -1,
        subset_seed: int = 42,
        grad_acc_steps: int = 1,
        drop_last: bool = True,
        **kwargs,
    ):
        if subset_size > 0:
            subset_size = (
                len(self.dset[split])
                if len(self.dset[split]) < subset_size
                else subset_size
            )
            dset = self.dset[split].shuffle(seed=subset_seed).select(range(subset_size))
        else:
            dset = self.dset[split]

        kwargs = {"batch_size": 32, "drop_last": drop_last} | kwargs
        assert (
            kwargs["batch_size"] % grad_acc_steps == 0
        ), "batch size must be divisible by gradient accumulation steps"
        kwargs["batch_size"] = kwargs["batch_size"] // grad_acc_steps

        if is_s2s:
            return self.s2s_loader(dset, *args, **kwargs)
        else:
            return self.clm_loader(dset, *args, **kwargs)

    def _tokenize_prompts(self, prompts):
        prompts = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )
        return prompts


class BoolQDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        max_seq_len: int = 256,
    ):
        dset = load_dataset("boolq")
        prompt = """Read the passage below and answer the question with the words 'true' or 'false'.

Passage: {passage}
Question: {question}
Answer (true or false):"""
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=False,
            boolean=True,
            max_seq_len=max_seq_len,
        )

    def clm_collate_fn(self, batch):
        prompts = [
            self.preamble.format(passage=e["passage"][:1024], question=e["question"])
            for e in batch
        ]
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([int(e["answer"]) for e in batch])
        targets = t.cat([self.label2target[c.item()] for c in classes])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch):
        prompts = [
            self.preamble.format(passage=e["passage"], question=e["question"])
            for e in batch
        ]
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([int(e["answer"]) for e in batch])
        targets = t.cat([self.label2target[c.item()] for c in classes])
        return prompts, targets, targets


boolq = BoolQDataset


class OBQADataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_seq_len: int = 512,
    ):
        dset = load_dataset("openbookqa", "main")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            4,
            prompt,
            add_space,
            numerical=False,
            max_seq_len=max_seq_len,
        )

    few_shot_preamble = """Return the abel of the correct answer for each question below.

The sun is responsible for
Choices:
A) puppies learning new tricks
B) children growing up and getting old
C) flowers wilting in a vase
D) plants sprouting, blooming and wilting
Answer: D

What doesn't eliminate waste?
A) plants
B) robots
C) mushrooms
D) bacteria
Answer: B

{question}
Choices:
{choices}
Answer:"""

    zero_shot_preamble = """Return the label of the correct answer for the question below.

Question: {question}
Chioces:
{choices}
Answer:"""

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            choices = "\n".join(
                [
                    f"{l}) {c}"
                    for l, c, in zip(e["choices"]["text"], e["choices"]["label"])
                ]
            )
            prompts.append(
                self.preamble.format(question=e["question_stem"], choices=choices)
            )
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([ord(e["answerKey"]) - ord("A") for e in batch])
        targets = t.cat([self.label2target[c.item()] for c in classes])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([ord(e["answerKey"]) - ord("A") for e in batch])
        targets = t.cat([self.label2target[c.item()] for c in classes])
        return prompts, targets, targets


obqa = OBQADataset


class ArcSplit(Enum):
    C = "ARC-Challenge"
    E = "ARC-Easy"


class ARCDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        name: ArcSplit = ArcSplit.E,
        add_space: bool = True,
        few_shot: bool = False,
        max_seq_len: int = 512,
    ):
        dset = load_dataset("ai2_arc", name)
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            5,
            prompt,
            add_space,
            numerical=False,
            max_seq_len=max_seq_len,
        )

    few_shot_preamble = """Return the label of the correct answer for each question below.

Which two body systems are directly involved in movement?
Choices:
A) muscular and skeletal
B) digestive and muscular
C) skeletal and respiratory
E) respiratory and digestive
Answer: A

{question}
Choices:
{choices}
Answer:"""

    zero_shot_preamble = """Return the label of the correct answer for the question below.

Question: {question}
Choices:
{choices}
Answer:"""

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            choices = "\n".join(
                [
                    f"{l}) {c}"
                    for l, c in zip(e["choices"]["text"], e["choices"]["label"])
                ]
            )
            prompts.append(
                self.preamble.format(question=e["question"], choices=choices)
            )
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        prompts = self._tokenize_prompts(prompts)
        classes_alpha = t.tensor([ord(e["answerKey"]) - ord("A") for e in batch])
        classes_num = []
        for e in batch:
            try:
                classes_num.append(int(e["answerKey"]) - 1)
            except:
                classes_num.append(-1)
        # classes_num = t.tensor([int(e["answerKey"]) - 1 for e in batch])
        classes = t.where(classes_alpha < 0, t.tensor(classes_num), classes_alpha)
        targets = t.cat([self.label2target[c.item()] for c in classes])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([ord(e["answerKey"]) - ord("A") for e in batch])
        targets = t.cat([self.label2target[c.item()] for c in classes])
        # just return the target token ids
        return prompts, targets, targets


arc = ARCDataset


class WinograndeSplit(Enum):
    XS = "winogrande_xs"
    S = "winogrande_s"
    M = "winogrande_m"
    L = "winogrande_l"
    XL = "winogrande_xl"


class WinograndeDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        name: WinograndeSplit = WinograndeSplit.S,
        add_space: bool = True,
        few_shot: bool = False,
        max_seq_len: int = 512,
    ):
        dset = load_dataset("winogrande", name, trust_remote_code=True)
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=False,
            max_seq_len=max_seq_len,
        )

    few_shot_preamble = """Return the label of the correct answer for each question below.

Adam put handwash only clothes in the washer but Aaron washed them by hand as _ was lazy.
Choices:
A) Adam
B) Aaron
Answer: A

Steven proudly showed Michael the mangoes he grew himself all this summer. _ is astonished.
Choices:
A) Stephen
B) Michael
Answer: B

{question}
Choices:
{choices}
Answer:"""

    zero_shot_preamble = """Return the label of the correct answer for the question below.

Question: {question}
Choices:
{choices}
Answer:"""

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            choices = f"A) {e['option1']}\nB) {e['option2']}"
            prompts.append(
                self.preamble.format(question=e["sentence"], choices=choices)
            )
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([int(e["answer"]) - 1 for e in batch])
        targets = t.cat([self.label2target[c.item()] for c in classes])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch):
        prompts = [e["sentence"] for e in batch]
        prompts = self._tokenize_prompts(prompts)
        targets = t.tensor([int(e["answer"]) - 1 for e in batch])
        return prompts, targets, targets


winogrande = WinograndeDataset


class CommonsenseQADataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = True,
        max_seq_len: int = 512,
    ):
        dset = load_dataset("commonsense_qa")
        super().__init__(
            dset,
            tokenizer,
            5,
            self.few_shot_preamble if few_shot else self.zero_shot_preamble,
            add_space,
            numerical=False,
            max_seq_len=max_seq_len,
        )

    # few-shot preamble
    few_shot_preamble = """Answer the questions below correctly.

Question: What do people aim to do at work?
Choices:
A) complete job
B) learn from each other
C) kill animals
D) wear hats
E) talk to each other
Answer: A

Question: Where do adults use glue sticks?
Choices:
A) classroom
B) desk drawer
C) at school
D) office
E) kitchen draw
Answer: D

Question: {question}
Choices:
{choices}
Answer:"""

    zero_shot_preamble = """Answer the multiple choice question below by returning the answer label (A to E)

Question: {question}
Choices:
{choices}
Answer:"""

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            choices = "\n".join(
                [
                    f"{l}) {c}"
                    for l, c in zip(e["choices"]["label"], e["choices"]["text"])
                ]
            )
            prompts.append(
                self.preamble.format(question=e["question"], choices=choices)
            )
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        prompts = self._tokenize_prompts(prompts)
        # targets are token ids of the correct answer
        spc = " " if self.add_space else ""
        targets = self.tokenizer(
            [f'{spc}{e["answerKey"]}' for e in batch],
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[:, -1]
        # classes are integers corresponding to the index of the correct answer
        base = ord("0") if self.numerical else ord("A")
        classes = t.tensor([ord(e["answerKey"]) - base for e in batch])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        prompts = self._tokenize_prompts(prompts)
        spc = " " if self.add_space else ""
        targets = self.tokenizer(
            [f'{spc}{e["answerKey"]}' for e in batch],
            return_tensors="pt",
            add_spcecial_tokens=False,
        ).input_ids[:, -1:]
        return prompts, targets, targets


cqa = CommonsenseQADataset


class CoLADataset(ClassificationDataset):
    def __init__(
        self, tokenizer: AutoTokenizer, add_space: bool = True, max_seq_len: int = 512
    ):
        dset = load_dataset("glue", "cola")
        super().__init__(
            dset, tokenizer, 2, self.preamble, add_space, max_seq_len=max_seq_len
        )

    preamble = """For each sentence below, indicate whether it is grammatically acceptable (1) or unacceptable (0).

Sentence: If you had eaten more, you would want less.
Answer: 1

Sentence: As you eat the most, you want the least.
Answer: 0

Sentence: {sentence}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [self.preamble.format(sentence=e["sentence"]) for e in batch]
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label2target[e["label"]] for e in batch])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch):
        prompts = [e["sentence"] for e in batch]
        prompts = self._tokenize_prompts(prompts)
        targets = t.tensor([e["label"] for e in batch])
        return prompts, targets, targets


cola = CoLADataset


class MNLIDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        max_seq_len: int = 512,
    ):
        dset = load_dataset("glue", "mnli")
        super().__init__(
            dset, tokenizer, 3, self.preamble, add_space, max_seq_len=max_seq_len
        )

    preamble = """For each premise below, indicate whether the hypothesis entails (0), is neutral towards (1) or contradicts (2) the premise.

Hypothesis: Buffet and a la carte available.
Premise: It has a buffet.
Answer: 0

Hypothesis: He had never felt better.
Premise: The medicine he had taken had worked well.
Answer: 1

Hypothesis: Oh, what a fool I feel!
Premise: I am beyond proud
Answer: 2

Hypothesis: {hypothesis}
Premise: {premise}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(hypothesis=e["hypothesis"], premise=e["premise"])
            for e in batch
        ]
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label2target[e["label"]] for e in batch])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch):
        prompts = [e["hypothesis"] + " " + e["premise"] for e in batch]
        prompts = self._tokenize_prompts(prompts)
        targets = t.tensor([e["label"] for e in batch])
        return prompts, targets, targets


mnli = MNLIDataset


class MRPCDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        max_seq_len: int = 512,
    ):
        dset = load_dataset("glue", "mrpc")
        super().__init__(
            dset, tokenizer, 2, self.preamble, add_space, max_seq_len=max_seq_len
        )

    preamble = """For each pair of sentences below, indicate whether the Sentence 1 is equivalent (1) or not equivalent (2) to the Sentence 2.

Sentence 1: Yucaipa owned Dominick's before selling the chain to Safeway in 1998 for $2.5 billion.
Sentence 2: Yucaipa bought Dominick's in 1995 for $693 million and sold it to Safeway for $1.8 billion in 1998.
Answer: 0

Sentence 1: Amrozi accused his brother, whom he called "the witness", of deliberately distorting his evidence.
Sentence 2: Referring to him as only "the witness", Amrozi accused his brother of deliberately distorting his evidence.
Answer: 1

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(sentence_1=e["sentence1"], sentence_2=e["sentence2"])
            for e in batch
        ]
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label2target[e["label"]] for e in batch])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch):
        prompts = [e["sentence1"] + " " + e["sentence2"] for e in batch]
        prompts = self._tokenize_prompts(prompts)
        targets = t.tensor([e["label"] for e in batch])
        return prompts, targets, targets


mrpc = MRPCDataset


class QNLIDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        max_seq_len: int = 512,
    ):
        dset = load_dataset("glue", "qnli")
        super().__init__(
            dset, tokenizer, 2, self.preamble, add_space, max_seq_len=max_seq_len
        )

    preamble = """For each sentence below, indicate whether it entails (0) or does not entail (1) the associated question.

Question: Which collection of minor poems are sometimes attributed to Virgil?
Sentence: A number of minor poems, collected in the Appendix Vergiliana, are sometimes attributed to him.
Answer: 0

Question: What was the highest order of species n land?
Sentence: The climate was much more humid than the Triassic, and as a result, the world was very tropical.
Answer: 1

Question: {question}
Sentence: {sentence}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(question=e["question"], sentence=e["sentence"])
            for e in batch
        ]
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label2target[e["label"]] for e in batch])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch):
        prompts = [e["question"] + " " + e["sentence"] for e in batch]
        prompts = self._tokenize_prompts(prompts)
        targets = t.tensor([e["label"] for e in batch])
        return prompts, targets, targets


qnli = QNLIDataset


class QQPDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        max_seq_len: int = 512,
    ):
        dset = load_dataset("glue", "qqp")
        super().__init__(
            dset, tokenizer, 2, self.preamble, add_space, max_seq_len=max_seq_len
        )

    preamble = """For each pair of questions below, indicate whether the first is a duplicate (1) or not a duplicate (0) of the first.

Question 1: How is air traffic controlled?
Question 2: How do you become an air traffic controller?
Answer: 0

Question 1: What are the coolest Android hacks and tricks you know?
Question 2: What are some cool hacks for Android phones?
Answer: 1

Question 1: {question_1}
Question 2: {question_2}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(question_1=e["question1"], question_2=e["question2"])
            for e in batch
        ]
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label2target[e["label"]] for e in batch])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch):
        prompts = [e["question1"] + " " + e["question2"] for e in batch]
        prompts = self._tokenize_prompts(prompts)
        targets = t.tensor([e["label"] for e in batch])
        return prompts, targets, targets


qqp = QQPDataset


class RTEDataset(ClassificationDataset):
    def __init__(
        self, tokenizer: AutoTokenizer, add_space: bool = True, max_seq_len: int = 512
    ):
        dset = load_dataset("glue", "rte")
        super().__init__(
            dset, tokenizer, 2, self.preamble, add_space, max_seq_len=max_seq_len
        )

    preamble = """For each pair of sentences below, indicate whether the second entails (0) or does not entail (1) the first.

Sentence 1: Edward VIII became King in January of 1936 and abdicated in December.
Sentence 2: King Edward VIII abdicated in December 1936.
Answer: 0

Sentence 1: No Weapons of Mass Destruction Found in Iraq Yet.
Sentence 2: Weapons of Mass Destruction Found in Iraq.
Answer: 1

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(sentence_1=e["sentence1"], sentence_2=e["sentence2"])
            for e in batch
        ]
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label2target[e["label"]] for e in batch])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch):
        prompts = [e["sentence1"] + " " + e["sentence2"] for e in batch]
        prompts = self._tokenize_prompts(prompts)
        targets = t.tensor([e["label"] for e in batch])
        return prompts, targets, targets


rte = RTEDataset


class SST2Dataset(ClassificationDataset):
    def __init__(
        self, tokenizer: AutoTokenizer, add_space: bool = True, max_seq_len: int = 512
    ):
        dset = load_dataset("glue", "sst2")
        super().__init__(
            dset, tokenizer, 2, self.preamble, add_space, max_seq_len=max_seq_len
        )

    preamble = """For each sentence below, indicate whether the sentiment is negative (0) or positive (1).

Sentence: a depressed fifteen-year-old 's suicidal poetry
Answer: 0

Sentence: the greatest musicians
Answer: 1

Sentence: {sentence}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [self.preamble.format(sentence=e["sentence"]) for e in batch]
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label2target[e["label"]] for e in batch])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch):
        prompts = [e["sentence"] for e in batch]
        prompts = self._tokenize_prompts(prompts)
        targets = t.tensor([e["label"] for e in batch])
        return prompts, targets, targets


sst2 = SST2Dataset


class WNLIDataset(ClassificationDataset):
    def __init__(
        self, tokenizer: AutoTokenizer, add_space: bool = True, max_seq_len: int = 512
    ):
        dset = load_dataset("glue", "wnli")
        super().__init__(
            dset, tokenizer, 2, self.preamble, add_space, max_seq_len=max_seq_len
        )

    preamble = """For each pair of sentences below, indicate whether the second entails (1) or does not entail (0) the first.

Sentence 1: Steve follows Fred's example in everything. He influences him hugely.
Sentence 2: Steve influences him hugely.
Answer: 0

Sentence 1: The police arrested all of the gang members. They were trying to stop the drug trade in the neighborhood.
Sentence 2: The police were trying to stop the drug trade in the neighborhood.
Answer: 1

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(sentence_1=e["sentence1"], sentence_2=e["sentence2"])
            for e in batch
        ]
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label2target[e["label"]] for e in batch])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch):
        prompts = [e["sentence1"] + " " + e["sentence2"] for e in batch]
        prompts = self._tokenize_prompts(prompts)
        targets = t.tensor([e["label"] for e in batch])
        return prompts, targets, targets


wnli = WNLIDataset

MMLUSplit = {
    "cs": [
        "college_computer_science",
        "high_school_computer_science",
        "computer_security",
        "machine_learning",
    ],
    # "cs" : ["abstract_algebra"],
    "eng": ["electrical_engineering"],
    "law": ["international_law", "jurisprudence", "professional_law"],
    "health": [
        "anatomy",
        "clinical_knowledge",
        "college_medicine",
        "human_aging",
        "nutrition",
        "professional_medicine",
        "virology",
    ],
    "chem": ["college_chemistry"],
    "bio": ["college_biology"],
    "phy": ["college_physics"],
}


class MMLUDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        name: str = "cs",
        add_space: bool = True,
        few_shot: bool = False,
        max_seq_len: int = 512,
    ):

        dset = load_dataset("cais/mmlu", "all")
        self.name = name

        df = pd.DataFrame(dset["test"])
        filtered_df = df[df["subject"].isin(MMLUSplit[name])]
        dset = datasets.Dataset.from_pandas(filtered_df)

        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            4,
            prompt,
            add_space,
            numerical=False,
            max_seq_len=max_seq_len,
        )

    few_shot_preamble = """Return the label of the correct answer for each question below.

Adam put handwash only clothes in the washer but Aaron washed them by hand as _ was lazy.
Choices:
A) Adam
B) Aaron
Answer: A

Steven proudly showed Michael the mangoes he grew himself all this summer. _ is astonished.
Choices:
A) Stephen
B) Michael
Answer: B

{question}
Choices:
{choices}
Answer:"""

    zero_shot_preamble = """Return the label of the correct answer for the question below.

Question: {question}
Choices:
{choices}
Answer:"""

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            # choices = "\n".join(e["choices"])
            choices = "\n".join(
                [f"{l}) {c}" for l, c, in zip(["A", "B", "C", "D"], e["choices"])]
            )
            prompts.append(
                self.preamble.format(question=e["question"], choices=choices)
            )
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([int(e["answer"]) for e in batch])
        targets = t.cat([self.label2target[c.item()] for c in classes])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch):
        prompts = [e["question"] for e in batch]
        prompts = self._tokenize_prompts(prompts)
        targets = t.tensor([int(e["answer"]) for e in batch])
        return prompts, targets, targets

    def loader(
        self,
        *args,
        is_s2s: bool = False,
        split: str = "test",
        subset_size: int = -1,
        subset_seed: int = 42,
        grad_acc_steps: int = 1,
        drop_last: bool = True,
        **kwargs,
    ):
        dset = self.dset
        kwargs = {"batch_size": 32, "drop_last": drop_last} | kwargs
        assert (
            kwargs["batch_size"] % grad_acc_steps == 0
        ), "batch size must be divisible by gradient accumulation steps"
        kwargs["batch_size"] = kwargs["batch_size"] // grad_acc_steps

        if is_s2s:
            return self.s2s_loader(dset, *args, **kwargs)
        else:
            return self.clm_loader(dset, *args, **kwargs)


mmlu = MMLUDataset


class LMDataset:
    """
    An abstract base dataset for autoregressive language modelling problems,
    where the main measure of success is the perplexity of the language model.
    """

    def __init__(
        self,
        dset,
        tokenizer,
        add_space: bool = False,
        max_seq_len: int = 512,
    ):
        """
        Args:
            dset: The loaded Dataset
            tokenizer: The model tokenizer
            preamble: Preamble for general pre-trained / 'CausalLM' models
            add_space: Add an explicit space suffix between preamble and answer tokens.
        """
        self.dset = dset
        self.add_space = add_space
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    @abstractmethod
    def s2s_collate_fn(self, batch):
        """Collate function for sequence to sequence models"""
        raise NotImplementedError

    def s2s_loader(self, dset: Dataset, *args, **kwargs) -> DataLoader:
        """Returns the dataloader for sequence to sequence models"""
        kwargs = {"batch_size": 32} | kwargs
        return t.utils.data.DataLoader(
            dset, collate_fn=self.s2s_collate_fn, *args, **kwargs
        )

    @abstractmethod
    def s2s_train_collate_fn_ori(self, batch):
        """Collate function for sequence to sequence models"""
        raise NotImplementedError

    def s2s_trainloader(self, dset: Dataset, *args, **kwargs) -> DataLoader:
        """Returns the dataloader for sequence to sequence models"""
        kwargs = {"batch_size": 32} | kwargs
        return t.utils.data.DataLoader(
            dset, collate_fn=self.s2s_train_collate_fn_ori, *args, **kwargs
        )

    def loader(
        self,
        *args,
        split: str = "train",
        subset_size: int = -1,
        **kwargs,
    ):
        if subset_size > 0:
            print(f"Using subset of size {subset_size}")
            dset = self.dset[split].select(range(subset_size))
        else:
            dset = self.dset[split]
        if split == "train":
            return self.s2s_trainloader(dset, *args, **kwargs)
        else:
            return self.s2s_loader(dset, *args, **kwargs)

    def _tokenize_prompts_batch(self, prompts):
        prompts = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        )
        return prompts

    def _tokenize_prompts(self, prompts):
        prompts = self.tokenizer(prompts, return_tensors="pt")
        return prompts


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class TriviaQADataset(LMDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        n_shot: int = 0,
        max_seq_len: int = 512,
        multianswer: bool = False,
    ):

        dset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
        self.n_shot = n_shot

        super().__init__(dset, tokenizer, add_space, max_seq_len=max_seq_len)

    zero_shot_preamble = """Answer the question below. Question: {question} Answer:"""
    sample_preamble = (
        """Answer the question below. Question: {question} Answer: {answer}"""
    )
    qa_preamble = """Question: {question} Answer:"""

    def _select_few_shot_samples(self, dset, n):
        indices = list(range(len(dset)))
        random.shuffle(indices)
        selected_indices = indices[:n]
        sampled_questions = [dset["question"][i] for i in selected_indices]
        sampled_answers = [
            normalize_answer(dset["answer"][i]["value"]) for i in selected_indices
        ]
        self.few_shot_samples = {
            "question": sampled_questions,
            "answer": sampled_answers,
        }

    def _generate_n_shot_preamble(self):
        example_str = ""
        for i in range(self.n_shot):
            example_str += f"Example: Question: {self.few_shot_samples['question'][i]} Answer: {normalize_answer(self.few_shot_samples['answer'][i])}. "

        return example_str + """Now, answer the question below. """

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            if self.n_shot > 0:
                self._select_few_shot_samples(self.dset["train"][:100], self.n_shot)
                prompt = self._generate_n_shot_preamble()
                prompts.append(prompt + self.qa_preamble.format(question=e["question"]))
            else:
                prompt = self.zero_shot_preamble
                prompts.append(prompt.format(question=e["question"]))

        return prompts

    def _format_samples(self, batch):
        prompts = []
        for e in batch:
            prompts.append(
                self.sample_preamble.format(
                    question=e["question"],
                    answer=normalize_answer(e["answer"]["value"]),
                )
            )
        return prompts

    def s2s_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        prompts = self._tokenize_prompts_batch(prompts)
        targets = self._tokenize_prompts_batch(
            [normalize_answer(e["answer"]["value"]) for e in batch]
        )
        targets_aliases = [
            self._tokenize_prompts_batch(
                [normalize_answer(a) for a in e["answer"]["aliases"]]
            )
            for e in batch
        ]
        return prompts, targets, targets_aliases

    # def s2s_train_collate_fn(self, batch):
    #     ignore_index = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False, return_tensors="pt").squeeze(0)
    #     prompts = self._format_samples(batch)
    #     tokenized_prompts = self._tokenize_prompts_batch(prompts)
    #     tokenized_targets = tokenized_prompts.copy()
    #     # Token ID for the string "Answer: " - assume you have a method to get this ID or it's a predefined constant
    #     answer_token_id_1 = self.tokenizer.encode(" Answer:", add_special_tokens=False, return_tensors="pt").squeeze(0)
    #     answer_token_id_2 = self.tokenizer.encode("Answer:", add_special_tokens=False, return_tensors="pt").squeeze(0)
    #     for i, prompt in enumerate(tokenized_prompts['input_ids']):
    #         # Find the position of "Answer: " in the tokenized prompt
    #         pos = -1
    #         for j in range(len(prompt)):
    #             if (prompt[j:j+len(answer_token_id_1)] == answer_token_id_1).all():
    #                 pos = j + len(answer_token_id_1)
    #                 break
    #             if (prompt[j:j+len(answer_token_id_2)] == answer_token_id_2).all():
    #                 pos = j + len(answer_token_id_2)
    #                 break

    #         # Set all tokens up to and including the position of "Answer: " to pad_token
    #         if pos != -1:
    #             tokenized_targets['input_ids'][i][:pos] = ignore_index.repeat(pos)

    #     return tokenized_prompts, tokenized_targets

    def s2s_train_collate_fn_ori(self, batch):
        prompts = self._format_samples(batch)
        tokenized_prompts = self._tokenize_prompts_batch(prompts)
        tokenized_targets = tokenized_prompts.copy()

        return tokenized_prompts, tokenized_targets


TriviaQA = TriviaQADataset


from datasets import concatenate_datasets


class AmbigQADataset(LMDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        n_shot: int = 0,
        max_seq_len: int = 512,
        multianswer: bool = True,
    ):
        dset = load_dataset("sewon/ambig_qa", "full")
        self.n_shot = n_shot

        super().__init__(dset, tokenizer, add_space, max_seq_len=max_seq_len)

        if multianswer:

            filtered_gt_1 = dset["validation"].filter(
                lambda example: len(example["nq_answer"]) > 1
            )
            filtered_eq_1 = dset["validation"].filter(
                lambda example: len(example["nq_answer"]) == 1
            )
            filtered_eq_1 = filtered_eq_1.select(range(200))
            dset["validation"] = concatenate_datasets([filtered_gt_1, filtered_eq_1])

    zero_shot_preamble = """Answer the question below. Question: {question} Answer:"""
    sample_preamble = (
        """Answer the question below. Question: {question} Answer: {answer}"""
    )
    qa_preamble = """Question: {question} Answer:"""

    def _select_few_shot_samples(self, dset, n):
        indices = list(range(len(dset)))
        random.shuffle(indices)
        selected_indices = indices[:n]
        sampled_questions = [dset["question"][i] for i in selected_indices]
        sampled_answers = [
            normalize_answer(dset["nq_answer"][i]) for i in selected_indices
        ]
        self.few_shot_samples = {
            "question": sampled_questions,
            "answer": sampled_answers,
        }

    def _generate_n_shot_preamble(self):
        example_str = ""
        for i in range(self.n_shot):
            example_str += f"Example: Question: {self.few_shot_samples['question'][i]} Answer: {normalize_answer(self.few_shot_samples['nq_answer'][i])}. "

        return example_str + """Now, answer the question below. """

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            if self.n_shot > 0:
                self._select_few_shot_samples(self.dset["train"][:100], self.n_shot)
                prompt = self._generate_n_shot_preamble()
                prompts.append(prompt + self.qa_preamble.format(question=e["question"]))
            else:
                prompt = self.zero_shot_preamble
                prompts.append(prompt.format(question=e["question"]))

        return prompts

    def _format_samples(self, batch):
        prompts = []
        for e in batch:
            prompts.append(
                self.sample_preamble.format(
                    question=e["question"], answer=normalize_answer(e["nq_answer"][0])
                )
            )
        return prompts

    def s2s_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        prompts = self._tokenize_prompts_batch(prompts)
        targets_aliases = [
            self._tokenize_prompts_batch([normalize_answer(a) for a in e["nq_answer"]])
            for e in batch
        ]
        return prompts, None, targets_aliases

    def s2s_train_collate_fn_ori(self, batch):
        prompts = self._format_samples(batch)
        tokenized_prompts = self._tokenize_prompts_batch(prompts)
        tokenized_targets = tokenized_prompts.copy()

        return tokenized_prompts, tokenized_targets


AmbigQA = AmbigQADataset


class OpenARCDataset(LMDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        n_shot: int = 0,
        max_seq_len: int = 512,
        multianswer: bool = True,
    ):
        dset = load_dataset("ai2_arc", "ARC-Easy")
        self.n_shot = n_shot

        super().__init__(dset, tokenizer, add_space, max_seq_len=max_seq_len)

    zero_shot_preamble = """Answer the question below. Question: {question} Answer:"""
    sample_preamble = (
        """Answer the question below. Question: {question} Answer: {answer}"""
    )
    qa_preamble = """Question: {question} Answer:"""

    def generate_answer_from_choices(self, choices, answer_key):
        index = choices["label"].index(answer_key)
        return choices["text"][index]

    def _select_few_shot_samples(self, dset, n):
        indices = list(range(len(dset)))
        random.shuffle(indices)
        selected_indices = indices[:n]
        sampled_questions = [dset["question"][i] for i in selected_indices]
        sampled_answers = [
            normalize_answer(dset["nq_answer"][i]) for i in selected_indices
        ]
        self.few_shot_samples = {
            "question": sampled_questions,
            "answer": sampled_answers,
        }

    def _generate_n_shot_preamble(self):
        example_str = ""
        for i in range(self.n_shot):
            example_str += f"Example: Question: {self.few_shot_samples['question'][i]} Answer: {normalize_answer(self.few_shot_samples['nq_answer'][i])}. "

        return example_str + """Now, answer the question below. """

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            if self.n_shot > 0:
                self._select_few_shot_samples(self.dset["train"][:100], self.n_shot)
                prompt = self._generate_n_shot_preamble()
                prompts.append(prompt + self.qa_preamble.format(question=e["question"]))
            else:
                prompt = self.zero_shot_preamble
                prompts.append(prompt.format(question=e["question"]))

        return prompts

    def _format_samples(self, batch):
        prompts = []
        for e in batch:
            prompts.append(
                self.sample_preamble.format(
                    question=e["question"],
                    answer=normalize_answer(
                        self.generate_answer_from_choices(e["choices"], e["answerKey"])
                    ),
                )
            )
        return prompts

    def s2s_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        prompts = self._tokenize_prompts_batch(prompts)
        targets = [
            self._tokenize_prompts_batch(
                [
                    normalize_answer(
                        self.generate_answer_from_choices(a["choices"], a["answerKey"])
                    )
                    for a in [e]
                ]
            )
            for e in batch
        ]
        return prompts, None, targets

    def s2s_train_collate_fn_ori(self, batch):
        prompts = self._format_samples(batch)
        tokenized_prompts = self._tokenize_prompts_batch(prompts)
        tokenized_targets = tokenized_prompts.copy()

        return tokenized_prompts, tokenized_targets


OpenARC = OpenARCDataset


import random
from datasets import load_dataset
from transformers import AutoTokenizer


class DollyDataset(LMDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        n_shot: int = 0,
        max_seq_len: int = 512,
        subset_name: str = None,
        random_seed: int = 1,
    ):
        dset = load_dataset("databricks/databricks-dolly-15k")
        self.n_shot = n_shot
        self.subset_name = subset_name

        super().__init__(dset, tokenizer, add_space, max_seq_len=max_seq_len)

        if subset_name is not None:
            subset_data = (
                dset["train"]
                .filter(lambda example: example["category"] == subset_name)
                .shuffle(seed=random_seed)
            )
            other_data = (
                dset["train"]
                .filter(lambda example: example["category"] != subset_name)
                .shuffle(seed=random_seed)
            )

            train_data = subset_data.select(range(1000))

            remaining_subset = subset_data.select(range(1000, len(subset_data)))
            id_evaluation_data = remaining_subset.select(range(100))

            print(f"{subset_name}: {len(subset_data)}")

            ood_evaluation_data = []
            unique_categories = set(other_data["category"])
            for category in unique_categories:
                category_data = other_data.filter(
                    lambda example: example["category"] == category
                )
                sampled_data = category_data.select(range(100))
                ood_evaluation_data.extend(sampled_data)
                print(f"{category}: {len(category_data)}")

            ood_evaluation_data = Dataset.from_list(ood_evaluation_data)

            dset["train"] = train_data
            dset["ood_evaluation"] = ood_evaluation_data
            dset["id_evaluation"] = id_evaluation_data

    general_train_preamble = (
        """{instruction}\n\nInput:\n{context}\n\nOutput:{response}"""
    )
    general_eval_preamble = """{instruction}\n\nInput:\n{context}\n\nOutput:"""

    def _select_few_shot_samples(self, dset, n):
        indices = list(range(len(dset)))
        random.shuffle(indices)
        selected_indices = indices[:n]
        sampled_questions = [dset["question"][i] for i in selected_indices]
        sampled_answers = [
            normalize_answer(dset["nq_answer"][i]) for i in selected_indices
        ]
        self.few_shot_samples = {
            "question": sampled_questions,
            "answer": sampled_answers,
        }

    def _generate_n_shot_preamble(self):
        example_str = ""
        for i in range(self.n_shot):
            example_str += f"Example: Question: {self.few_shot_samples['question'][i]} Answer: {normalize_answer(self.few_shot_samples['nq_answer'][i])}. "

        return example_str + """Now, answer the question below. """

    def _format_train_prompts(self, batch):
        prompts = []
        for e in batch:
            prompt = self.general_train_preamble
            prompts.append(
                prompt.format(
                    instruction=e["instruction"],
                    context=e["context"],
                    response=e["response"],
                )
            )

        return prompts

    def _format_eval_prompts(self, batch):
        prompts = []
        for e in batch:
            prompt = self.general_eval_preamble
            prompts.append(
                prompt.format(instruction=e["instruction"], context=e["context"])
            )

        return prompts

    def s2s_collate_fn(self, batch):
        prompts = self._format_eval_prompts(batch)
        prompts = self._tokenize_prompts_batch(prompts)
        targets = self._tokenize_prompts_batch([e["response"] for e in batch])
        return prompts, targets, None

    def s2s_train_collate_fn_ori(self, batch):
        prompts = self._format_train_prompts(batch)
        tokenized_prompts = self._tokenize_prompts_batch(prompts)
        tokenized_targets = self._tokenize_prompts_batch([e["response"] for e in batch])

        return tokenized_prompts, tokenized_targets


Dolly = DollyDataset

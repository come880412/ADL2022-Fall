from inspect import TPFLAGS_IS_ABSTRACT
from random import sample
from typing import List, Dict
import numpy as np

from torch.utils.data import Dataset
import torch

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        text_tokens = [batch["text"].split() for batch in samples]
        text_encoded = self.vocab.encode_batch(text_tokens, self.max_len)
        try:
            label = [self.label2idx(batch["intent"]) for batch in samples]
            label = torch.from_numpy(np.array(label))
        except:
            label = None
        id = [batch["id"] for batch in samples]

        text_encoded = torch.from_numpy(np.array(text_encoded))
        data = {"text":text_encoded, "label":label, "id":id}
        return data

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def label2idx(self, tags: list):
        label_tag = []
        for batch in range(len(tags)):
            tmp_tag = []
            for i in range(self.max_len):
                if i < len(tags[batch]):
                    tmp_tag.append(int(self.label_mapping[tags[batch][i]]))
                else:
                    tmp_tag.append(-1)
            label_tag.append(tmp_tag)
        return label_tag

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        batch_len = []
        text_tokens = [batch["tokens"] for batch in samples]
        for token in text_tokens:
            batch_len.append(len(token))
        text_encoded = self.vocab.encode_batch(text_tokens, self.max_len)
        try:
            tags = [batch["tags"] for batch in samples]
            tags = self.label2idx(tags)
            tags = torch.from_numpy(np.array(tags))
        except:
            tags = None

        text_encoded = torch.from_numpy(np.array(text_encoded))
        id = [batch["id"] for batch in samples]
        data = {"tokens":text_encoded, "tags":tags, "batch_len": batch_len, "id":id}

        return data

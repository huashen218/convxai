from typing import Dict, List, Optional
import logging

import os
import json
from allennlp.data import Tokenizer
from overrides import overrides
from nltk.tree import Tree


from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.common.checks import ConfigurationError

from pathlib import Path
from itertools import chain
import os.path as osp
import tarfile
import numpy as np
import math


logger = logging.getLogger(__name__)

TRAIN_VAL_SPLIT_RATIO = 0.9
        


def get_label(p):
    if "background" in p:
        return "0"
    elif "purpose" in p:
        return "1"
    elif "method" in p:
        return "2"
    elif "finding" in p:
        return "3"
    elif "other" in p:
        return "4"



def clean_text(text, special_chars=["\n", "\t"]):
    for char in special_chars:
        text = text.replace(char, " ")
    return text



@DatasetReader.register("coda19")
class Coda19DatasetReader(DatasetReader):


    def __init__(self, predictor,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._tokenizer = predictor.tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or \
                {"tokens": SingleIdTokenIndexer()}
        self.random_seed = 0 # numpy random seed


    def get_inputs(self, data_dir, phrase="train", return_labels = False):
        filenames = [f for f in os.listdir(os.path.join(data_dir, phrase)) if ".swp" not in f and ".json" in f]
        strings = []
        labels = []
        for filename in filenames:
            with open(os.path.join(data_dir, phrase, filename), 'r', encoding='utf-8') as infile:
                data = json.load(infile)
                strings.extend([
                    segment["segment_text"]
                    for paragraph in data["abstract"]
                    for sent in paragraph["sentences"]
                    for segment in sent
                ])
                labels.extend([
                    segment["crowd_label"]
                    for paragraph in data["abstract"]
                    for sent in paragraph["sentences"]
                    for segment in sent
                ])
        if return_labels:
            return strings, labels
        return strings 


    def text_to_instance(
            self, string: str, label:str = None) -> Optional[Instance]:
        tokens = self._tokenizer.tokenize(string)
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)


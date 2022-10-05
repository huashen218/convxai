import os
import json
from typing import Dict, List, Optional
import logging

from nltk.tree import Tree
from overrides import overrides
from allennlp.data import Tokenizer
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
    assert "pos" in p or "neg" in p
    return "1" if "pos" in p else "0"


def clean_text(text, special_chars=["\n", "\t"]):
    for char in special_chars:
        text = text.replace(char, " ")
    return text



@DatasetReader.register("diversity")
class DiversityDatasetReader(DatasetReader):

    TAR_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz' 
    TRAIN_DIR = 'aclImdb/train'
    TEST_DIR = 'aclImdb/test'
    

    def __init__(self, predictor,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 **kwargs) -> None:

        super().__init__(**kwargs)

        self._tokenizer = predictor.tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or \
                {"tokens": SingleIdTokenIndexer()}

        self.random_seed = 0 # numpy random seed
        # self.predictor = predictor


    def get_path(self, file_path):
        tar_path = cached_path(self.TAR_URL)
        tf = tarfile.open(tar_path, 'r')
        cache_dir = Path(osp.dirname(tar_path))
        if not (cache_dir / self.TRAIN_DIR).exists() and \
                not (cache_dir / self.TEST_DIR).exists():
            tf.extractall(cache_dir)

        if file_path == 'train':
            pos_dir = osp.join(self.TRAIN_DIR, 'pos')
            neg_dir = osp.join(self.TRAIN_DIR, 'neg')
            path = chain(
                    Path(cache_dir.joinpath(pos_dir)).glob('*.txt'), 
                    Path(cache_dir.joinpath(neg_dir)).glob('*.txt'))
        elif file_path in ['train_split', 'dev_split']:
            pos_dir = osp.join(self.TRAIN_DIR, 'pos')
            neg_dir = osp.join(self.TRAIN_DIR, 'neg')
            path = chain(
                    Path(cache_dir.joinpath(pos_dir)).glob('*.txt'), 
                    Path(cache_dir.joinpath(neg_dir)).glob('*.txt'))
            path_lst = list(path)
            np.random.shuffle(path_lst)
            num_train_strings = math.ceil(
                    TRAIN_VAL_SPLIT_RATIO * len(path_lst))
            train_path, path_lst[:num_train_strings]
            val_path = path_lst[num_train_strings:]
            path = train_path if file_path == "train" else val_path
        elif file_path == 'test':
            pos_dir = osp.join(self.TEST_DIR, 'pos')
            neg_dir = osp.join(self.TEST_DIR, 'neg')
            path = chain(
                    Path(cache_dir.joinpath(pos_dir)).glob('*.txt'), 
                    Path(cache_dir.joinpath(neg_dir)).glob('*.txt'))
        elif file_path == "unlabeled":
            unsup_dir = osp.join(self.TRAIN_DIR, 'unsup')
            path = chain(Path(cache_dir.joinpath(unsup_dir)).glob('*.txt'))
        else:
            raise ValueError(f"Invalid option for file_path.")
        return path
    

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



    @overrides
    def _read(self, file_path):
        np.random.seed(self.random_seed)
        tar_path = cached_path(self.TAR_URL)
        tf = tarfile.open(tar_path, 'r')
        cache_dir = Path(osp.dirname(tar_path))
        if not (cache_dir / self.TRAIN_DIR).exists() and \
                not (cache_dir / self.TEST_DIR).exists():
            tf.extractall(cache_dir)
        path = self.get_path(file_path)
        for p in path:
            label = get_label(str(p))
            yield self.text_to_instance(
                    clean_text(p.read_text(), special_chars=["<br />", "\t"]), 
                    label)


    def text_to_instance(
            self, string: str, label:str = None) -> Optional[Instance]:
        tokens = self._tokenizer.tokenize(string)
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)

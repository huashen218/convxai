# from transformers import T5Tokenizer, T5Model, T5Config
# from transformers import T5ForConditionalGeneration, T5TokenizerFast
# from allennlp.predictors import Predictor, TextClassifierPredictor
# from allennlp_models.classification import StanfordSentimentTreeBankDatasetReader
# from allennlp.data.tokenizers import PretrainedTransformerTokenizer
# from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

# import csv
# import heapq
# import sys
# import operator
# from tqdm import tqdm
# import re
# import nltk
# import warnings
# import argparse
# import pandas as pd
# import numpy as np
# import random
# import time

# import json



import os
import torch
import logging

from convxai.xai_models.trainers.explainers.counterfactual_explainer.src.utils import *
from convxai.xai_models.trainers.explainers.counterfactual_explainer.src.edit_finder import EditFinder, EditEvaluator, EditList
from convxai.xai_models.trainers.explainers.counterfactual_explainer.src.stage_two import load_models



logger = logging.getLogger("my-logger")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT)
logger.setLevel(logging.INFO)



class CounterfactualExplainer(object):
    
    def __init__(self):
        super().__init__()
        self.args = get_args("stage2")
        self.editor, self.predictor = load_models(self.args)
        self.edit_evaluator = EditEvaluator()
        self.edit_finder = EditFinder(self.predictor, self.editor, 
                beam_width=self.args.search.beam_width, 
                max_mask_frac=self.args.search.max_mask_frac,
                search_method=self.args.search.search_method,
                max_search_levels=self.args.search.max_search_levels)



    def generate_counterfactual(self, input_text, contrast_label):
        edited_list = self.edit_finder.minimally_edit(input_text, 
                                                        contrast_pred_idx_input = contrast_label, ### contrast_pred_idx specifies which label to use as the contrast. Defaults to -2, i.e. use label with 2nd highest pred prob.
                                                        max_edit_rounds=self.args.search.max_edit_rounds, 
                                                        edit_evaluator=self.edit_evaluator, 
                                                        max_length = int(self.args.model.model_max_length))

        torch.cuda.empty_cache()
        sorted_list = edited_list.get_sorted_edits() 
        print("===>>>counterfactual output - sorted_list:", sorted_list)

        if len(sorted_list) > 0:
            output = {
                "original_input": input_text,
                "counterfactual_input": sorted_list[0]['edited_editable_seg'],
                "counterfactual_label": sorted_list[0]['edited_label'],
                "counterfactual_confidence": sorted_list[0]['edited_contrast_prob'],
            }
        else:
            output = []
        return output
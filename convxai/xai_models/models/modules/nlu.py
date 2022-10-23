#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code supports the web server of the ConvXAI system.
# Copyright (c) Hua Shen 2022.
#

import torch
import torch.nn as nn
from typing import List
from transformers import BertTokenizer, BertModel
from convxai.utils import *



XAI_User_Intents = ["meta-data",
                    "meta-model",
                    "quality-score",
                    "aspect-distribution",
                    "confidence",
                    "example",
                    "attribution",
                    "counterfactual",
                    "sentence-length",
                    "ai-comment-global",
                    "ai-comment-instance"
                ]



class XAI_NLU_Module(nn.Module):
    r"""Construct the Natural Language Understanding module for user explanation requests.
    """
    def __init__(self, configs, nlu_algorithm = "rule_based"):
        super(XAI_NLU_Module, self).__init__()
        self.configs = configs
        self.nlu_algorithm = nlu_algorithm

        if self.nlu_algorithm == "scibert_classifier":
            self.tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
            self.model = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")
            self.question_embeddings, self.labels  = h5_load(self.configs['conversational_xai']['nlu_question_embedding_dir'], [
                                            "question_embeddings", "labels"],  verbose=True)
            self.question_embeddings = torch.from_numpy(self.question_embeddings)
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)


        self.user_intent_keywords = {
                    "meta-data": ["data", "dataset", "datasets"],
                    "meta-model": ["model","models"],
                    "quality-score": ["quality","style","scores", "score"],
                    "aspect-distribution": ["label","labels", "aspect", "aspects", "structure", "structures"],
                    "confidence": ["confidence","confident"],
                    "example": ["similar", "example", "examples"],
                    "attribution": ["important", "features", "attributions","feature", "attribution", "word", "words" ],
                    "counterfactual": ["different", "counterfactual", "prediction", "input", "revise"],
                    "sentence-length": ["length", "lengths"],
                    "ai-comment-global": ["show me model and data"],
                    "ai-comment-instance": ["explain this sentence", "explain the sentence", "explain sentence review", "explain this review", "explain the review"]
        }



    def nlu_rule_with_keyword(self, input_string, keyword_list) -> bool:

        # input_string = input_string.replace("?", "") 
        # input_list = input_string.split(" ")
        # return len(list(set(input_list).intersection(set(keyword_list)))) > 0

        input_string = input_string.replace("?", "") 
        keyphrases = [key for key in keyword_list if input_string.split(key) > 2 ]
        return len(keyphrases) > 0


    def intent_classification(self, input):
        r"""Using an intent classifier to classify the user input into different intents.
        Input: user_question sentences.
        Output: the class of user intent, e.g., XAI_User_Intents list.
        """

        input_string = input if type(input) == str else input[0]    #### only explain the first sentence of input
        inputs = self.tokenizer(input_string, return_tensors="pt")
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state[:, 0, :]
        similarity = self.cos(last_hidden_states, self.question_embeddings)

        ###### Strategy: choose the top one.   Alternative Strategy: choose the top five and choose majority vote ######  
        top_index = torch.argsort(similarity)[-1:]
        intent = self.labels[top_index]

        return intent.decode("utf-8") 





    def forward(self, input: List[str]) -> List[str]:
        r"""Rule-based NLU: The XAI_NLU module aims to parse the user input utterance and extract its intent and slot-fillers.
        Args:
            input (`List[str]`):
                List of user utterances.
        """
        input = str(input).strip().lower()

        ###### Rule-based user intents ###### 
        if self.nlu_algorithm == "rule_based":

            if self.nlu_rule_with_keyword(input, self.user_intent_keywords['meta-data']):
                return XAI_User_Intents[0]

            elif self.nlu_rule_with_keyword(input, self.user_intent_keywords['meta-model']):
                return XAI_User_Intents[1]

            elif self.nlu_rule_with_keyword(input, self.user_intent_keywords['quality-score']):
                return XAI_User_Intents[2]

            elif self.nlu_rule_with_keyword(input, self.user_intent_keywords['aspect-distribution']):
                return XAI_User_Intents[3]

            elif self.nlu_rule_with_keyword(input, self.user_intent_keywords['confidence']):
                return XAI_User_Intents[4]

            elif self.nlu_rule_with_keyword(input, self.user_intent_keywords['example']):
                return XAI_User_Intents[5]

            elif self.nlu_rule_with_keyword(input, self.user_intent_keywords['attribution']):
                return XAI_User_Intents[6]

            elif self.nlu_rule_with_keyword(input, self.user_intent_keywords['counterfactual']):
                return XAI_User_Intents[7]

            elif self.nlu_rule_with_keyword(input, self.user_intent_keywords['sentence-length']):
                return XAI_User_Intents[8]

            elif self.nlu_rule_with_keyword(input, self.user_intent_keywords['ai-comment-global']):
                return XAI_User_Intents[9]

            elif self.nlu_rule_with_keyword(input, self.user_intent_keywords['ai-comment-instance']):
                return XAI_User_Intents[10]

            else:
                return "none"


        ###### Using an Intent Classifier ######
        if self.nlu_algorithm == "scibert_classifier":
            user_intent = self.intent_classification(input)
            return user_intent

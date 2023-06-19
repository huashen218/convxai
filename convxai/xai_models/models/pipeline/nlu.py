#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code supports the web server of the ConvXAI system.
# Copyright (c) Hua Shen 2023.
#

import torch
from typing import List
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification
)
from convxai.utils import *

import pdb

# XAI_User_Intent_Map = {
#     0: '[counterfactual prediction]',
#     1: '[data statistics]',
#     2: '[important words]',
#     3: '[model description]',
#     4: '[other]',
#     5: '[prediction confidence]',
#     6: '[similar examples]',
# }


XAI_User_Intent_Map = {
    0: '[counterfactual prediction]',
    1: '[data statistics]',
    2: '[important words]',
    3: '[label distribution]',
    4: '[model description]',
    5: '[other]',
    6: '[prediction confidence]',
    7: '[quality score]',
    8: '[sentence length]',
    9: '[similar examples]',
    10: '[xai tutorial]',
}


class XAI_NLU_Module:
    """The XAI_NLU_Module inputs the user's XAI question and output the XAI intent corresponding to a specific XAI algorithm.
    """

    def __init__(
        self,
        model_name_or_path: str,
        intent_threshold: float = 0.3  # This is a very important threshold variable to filter out '[other]' type
    ) -> None:
        self.config = AutoConfig.from_pretrained(model_name_or_path,
            num_labels=len(XAI_User_Intent_Map.keys())
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=self.config
        )
        
        # Note: you can adjust the intent_threshold to convert between pre-trained questions v.s. chatgpt s
        self.intent_threshold = intent_threshold

        # print("===>>>self.config.id2label", self.config.id2label)
        assert self.config.id2label == XAI_User_Intent_Map, "Please check the XAI Intent Classifier's label mapping with the 'id2label' variable."

    def _normalize(self, text: str) -> str:
        return text.strip().lower()

    def __call__(self, text: List[str]) -> List[str]:
        """
        Run the user XAI question classification.

        Args:
            text: The list of user input of XAI questions.

        Returns:
            The list of intent for all user XAI questions.
        """

        # Normalize the text of user input
        if isinstance(text, str):
            text = [text]
        text = [self._normalize(t) for t in text]

        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_position_embeddings,
        )
        

        # Run the SequenceClassification model
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        print("====>>>torch.max(probs).item()", torch.max(probs).item())

        # Convert the predictions to labels
        if torch.max(probs).item() >= self.intent_threshold:
            labels = [XAI_User_Intent_Map[pred.item()] for pred in preds]
        else:
            labels = ["[other]"]
        return labels


# Quick test
if __name__ == "__main__":

    # setting
    model_name_or_path = "/data/data/hua/workspace/projects/convxai/checkpoints/intent_model/intent-deberta-v3-base"

    # Load the model
    model = XAI_NLU_Module(model_name_or_path)

    # Test the model
    test_input = "What is the confidence of the prediction?"
    print("The input is: ", test_input)
    print("The intent is: ", model([test_input]))

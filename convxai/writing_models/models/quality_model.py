#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code supports the web server of the ConvXAI system.
# Copyright (c) Hua Shen 2022.
#


import os
import math
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from convxai.utils import *
from convxai.writing_models.dataloaders import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###### Quality Model Output Mapping ######
levels = [35, 52, 74, 116]

def quality_model_label_mapping(x): return \
    "quality1" if x > levels[3] else (
    "quality2" if x >= levels[2] and x < levels[3] else (
        "quality3" if x >= levels[1] and x < levels[2] else (
            "quality4" if x >= levels[0] and x < levels[1] else
            "quality5"))
)


class QualityModel(object):

    def __init__(self, saved_model_dir = "gpt2"):
        config = AutoConfig.from_pretrained(saved_model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(saved_model_dir, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            saved_model_dir,
            from_tf=bool(".ckpt" in saved_model_dir),
            config=config,
        ).to(device)


    def inference(self, inputs):
        """Generate Predictions for Users' Writing Inputs.
        """
        inputs = paragraph_segmentation(inputs)   ### convert inputs into fragments
        pred_outs = self.generate_prediction(inputs)  ### make predictions
        outputs = {
            "inputs": inputs,
            "outputs": pred_outs # pred_outs = (perplexities)
        }
        return outputs


    def generate_prediction(self, eval_data):
        self.model.eval()
        perplexities = []
        with torch.no_grad():
            for input_text in eval_data:
                inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
                inputs['labels'] = inputs['input_ids']
                outputs = self.model(**inputs)
                loss = outputs.loss
                perplexities.append(math.exp(loss))
        return (perplexities)



    def generate_confidence(self, inputs):
        eval_data = paragraph_segmentation(inputs)   ### convert inputs into fragments
        self.model.eval()
        perplexities = []
        probability = []
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for input_text in eval_data:
                inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
                inputs['labels'] = inputs['input_ids']
                outputs = self.model(**inputs)
                loss, y_pred = outputs[0:2]
                perplexities.append(math.exp(loss))
                prob = torch.max(softmax(y_pred), dim=1)[0]
                probability.append(prob.cpu().numpy())
        return (perplexities, probability)




    def generate_embeddings(self, inputs, eval_data = None):
        eval_data = paragraph_segmentation(inputs)
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for input_text in eval_data:
                inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
                inputs['labels'] = inputs['input_ids']
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs[3][-1] ### shape = torch.Size([1, 1024, 768]))
                embeddings.append(np.sum(hidden_states.detach().cpu().numpy(), axis=1))
        embeddings = np.concatenate(embeddings)
        return embeddings  ### shape = (7, 768)



    def extract(self, train_dataloader):
        all_embeddings = []
        self.model.eval()
        for step, batch in enumerate(tqdm(train_dataloader)):  ## input shape = torch.Size([1, 1024])
            outputs = self.model(**batch, output_hidden_states=True)
            """outputs = (loss, logits = shape = torch.Size([1, 1024, 50257]), past_key_values, hidden_states shape = tuple (3) -> outputs: torch.Size([1, 1024, 768]))
            """
            embeddings = torch.sum(outputs[3][-1], dim=1)  ### shape = torch.Size([1, 1024, 768]))
            all_embeddings.append(embeddings.detach().cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code supports the web server of the ConvXAI system.
# Copyright (c) Hua Shen 2022.
#


import os
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils import data
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)
from convxai.utils import *
from convxai.writing_models.dataloaders import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###### Diversity Model Output Mapping ######
diversity_model_label_mapping = {
    0: "background",
    1: "purpose",
    2: "method",
    3: "finding",
    4: "other",
}

label_mapping = {
    "background": 0,
    "purpose": 1,
    "method": 2, 
    "finding": 3,
    "other": 4,
}


class DiversityModel(object):
    def __init__(self, saved_model_dir=None):
        super(DiversityModel, self).__init__()
        if saved_model_dir is not None:
            config = AutoConfig.from_pretrained(saved_model_dir)
            config.num_labels = 5
            self.model = AutoModelForSequenceClassification.from_pretrained(
                saved_model_dir, config=config).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(saved_model_dir)
        else:
            config = AutoConfig.from_pretrained(
                'allenai/scibert_scivocab_uncased')
            config.num_labels = 5
            self.model = AutoModelForSequenceClassification.from_pretrained(
                'allenai/scibert_scivocab_uncased', config=config).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                'allenai/scibert_scivocab_uncased')

    def inference(self, inputs):
        """Generate Predictions for Users' Writing Inputs.
        """
        inputs = paragraph_segmentation(
            inputs)  # convert inputs into fragments
        eval_dataloader = self.load_infer_data(
            inputs)  # convert inputs into dataloader
        pred_outs = self.generate_prediction(eval_dataloader)
        outputs = {
            "inputs": inputs,
            "outputs": pred_outs  # pred_outs = (predict, probability)
        }
        return outputs

    def load_infer_data(self, x_text, batch_size=32):
        feature = Feature(tokenizer=self.tokenizer)
        x_text = feature.extract(x_text[:])
        test_dataset = PredDataset(x_text)
        test_dataloader = data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return test_dataloader

    def generate_prediction(self, eval_data):
        self.model.eval()
        predict = []
        probability = []
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for _, (x_batch) in enumerate(eval_data, 1):  # why start from 1?
                x_batch = x_batch.to(device)
                y_batch = torch.tensor([1]*x_batch.size(0)).to(device)
                outputs = self.model(x_batch, labels=y_batch)
                loss, y_pred = outputs[0:2]
                prob = torch.max(softmax(y_pred), dim=1)[0]
                y_pred = torch.argmax(y_pred, dim=1)
                predict.append(y_pred.cpu().numpy())
                probability.append(prob.cpu().numpy())
        predict = np.hstack(predict)
        probability = np.hstack(probability)
        return (predict, probability)

    def generate_confidence(self, inputs):
        # convert inputs into fragments
        inputs = paragraph_segmentation(inputs)
        # convert inputs into dataloader
        eval_data = self.load_infer_data(inputs)
        self.model.eval()
        predict = []
        probability = []
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for _, (x_batch) in enumerate(eval_data, 1):  # why start from 1?
                x_batch = x_batch.to(device)
                y_batch = torch.tensor([1]*x_batch.size(0)).to(device)
                outputs = self.model(x_batch, labels=y_batch)
                loss, y_pred = outputs[0:2]
                prob = torch.max(softmax(y_pred), dim=1)[0]
                y_pred = torch.argmax(y_pred, dim=1)
                predict.append(y_pred.cpu().numpy())
                probability.append(prob.cpu().numpy())
        predict = np.hstack(predict)
        probability = np.hstack(probability)
        return (predict, probability)

    def generate_embeddings(self, inputs):
        # convert inputs into fragments
        inputs = paragraph_segmentation(inputs)
        # convert inputs into dataloader
        eval_data = self.load_infer_data(inputs)
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for _, (x_batch) in enumerate(eval_data, 1):
                x_batch = x_batch.to(device)
                y_batch = torch.tensor([1]*x_batch.size(0)).to(device)
                outputs = self.model(
                    x_batch, labels=y_batch, output_hidden_states=True)
                # hidden_states shape = torch.Size([7, 102, 768])
                hidden_states = outputs[2][-1]
                embeddings.append(hidden_states.detach().cpu().numpy())
        # x_train_embeddings = (674, 768)
        embeddings = np.sum(np.concatenate(embeddings), axis=1)
        return embeddings  # shape = (7, 102, 768)

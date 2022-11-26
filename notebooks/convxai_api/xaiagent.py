#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code supports the web server of the ConvXAI system.
# Copyright (c) Hua Shen 2022.
#

import os
import json
import torch
from IPython.core.display import display, HTML

from .modules import *
from convxai.utils import parse_system_config_file
from convxai.writing_models.models import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvXAI(object):
    """The conversational XAI universal API class."""

    def __init__(self, intent_detection_algorithm="rule_based", model=None) -> None:
        # # ****** Load config file ****** #
        self.configs = parse_system_config_file()
        model  = DiversityModel(saved_model_dir=self.configs["scientific_writing"]["diversity_model_dir"]) if model is None else model
        # ****** Set up Convxai modules ****** #
        self.nlu = XAI_NLU_Module(self.configs, intent_detection_algorithm)
        self.explainer = Model_Explainer(model)
        self.nlg = XAI_NLG_Module()


    def interact_single_turn(self, user_input, explained_sentence, target_label, target_conference="CHI", visualize=False):
        user_intent = self.nlu(user_input)
        print(f"Detected User Input = {user_intent}")
        explanation_dict = self.explainer.generate_explanation(user_intent, explained_sentence, target_label, target_conference)
        response = self.nlg(user_intent, explanation_dict)
        if visualize:
            self._visualize_single_turn_dialog(user_input, response)
        return response


    def _visualize_single_turn_dialog(self, user_input, response):
        print("======> Conversational XAI Demonstration <======")
        print(f"User: {user_input}")
        display(HTML("ConvXAI: " + response))

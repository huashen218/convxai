#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code supports the web server of the ConvXAI system.
# Copyright (c) Hua Shen 2023.
#

import torch.nn as nn


class XAI_NLG_Module(nn.Module):
    """The XAI_NLG_Module input the output of the AI_Explainers and put them into the natural sentence templates to send to end users as conversational reponses.
    """

    def __init__(self):
        super(XAI_NLG_Module, self).__init__()

    def forward(self, explanations):
        return explanations

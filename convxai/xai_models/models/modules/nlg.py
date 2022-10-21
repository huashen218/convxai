#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code supports the web server of the ConvXAI system.
# Copyright (c) Hua Shen 2022.
#

import torch.nn as nn


class XAI_NLG_Module(nn.Module):

    def __init__(self):
        super(XAI_NLG_Module, self).__init__()
        """Template-based conversational XAI generation.
        """

    def forward(self, explanations):
        return explanations

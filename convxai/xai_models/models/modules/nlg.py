import torch
import torch.nn as nn
import torch.nn.functional as F



class XAI_NLG_Module(nn.Module):

    def __init__(self):
        super(XAI_NLG_Module, self).__init__()
        """For example-based explanations.
        Generate and load all training data embeddings.
        """


    def forward(self, explanations):
        return explanations

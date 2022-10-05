import os
import json
import torch.nn as nn
import torch
from typing import List, Optional, Tuple
from transformers import BertTokenizer, BertModel

# from convxai.xai_models.utils import h5_load


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')


XAI_User_Intents = ["meta-data",
                    "meta-model",
                    "quality-score",
                    "aspect-distribution",
                    "confidence",
                    "example",
                    "attribution",
                    "counterfactual",
                    "sentence-length"
                ]



import os
import torch
import pandas as pd
import numpy as np
import json
import h5py
from transformers import BertTokenizer, BertModel


def h5_load(filename, data_list, dtype=None, verbose=False):
    with h5py.File(filename, 'r') as infile:
        data = []
        for data_name in data_list:
            if dtype is not None:
                temp = np.empty(infile[data_name].shape, dtype=dtype)
            else:
                temp = np.empty(infile[data_name].shape, dtype=infile[data_name].dtype)
            infile[data_name].read_direct(temp)
            data.append(temp)
          
        if verbose:
            print("\n".join(
                "{} = {} [{}]".format(data_name, str(real_data.shape), str(real_data.dtype))
                for data_name, real_data in zip(data_list, data)
            ))
            print()
        return data




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

            self.question_embeddings, self.labels  = h5_load(self.configs['nlu_question_embedding_dir'], [
                                            "question_embeddings", "labels"],  verbose=True)

            self.question_embeddings = torch.from_numpy(self.question_embeddings)
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)



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
        ### TODO: Add others ###
        
        ###### Strategy: choose the top one ######  
        ###### Alternative Strategy: choose the top five and choose majority vote ######   
        top_index = torch.argsort(similarity)[-1:]

        intent = self.labels[top_index]
        print(f"===>>> NLU Intern = {intent} <<<====")
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

            input = input.replace("?", "") 
            input_list = input.split(" ")

            
            if len(list(set(input_list).intersection(set(["data", "dataset", "datasets"])))) > 0:  # "statistics"
                user_intent = XAI_User_Intents[0]
                return user_intent


            elif len(list(set(input_list).intersection(set(["confidence","confident"])))) > 0:
                return XAI_User_Intents[4]


            elif len(list(set(input_list).intersection(set(["similar", "example", "examples"])))) > 0:
                return XAI_User_Intents[5]


            elif len(list(set(input_list).intersection(set(["important", "features", "attributions","feature", "attribution", "word", "words" ])))) > 0:
                return XAI_User_Intents[6]


            elif len(list(set(input_list).intersection(set(["different", "counterfactual", "prediction", "input", "revise"])))) > 0:
                return XAI_User_Intents[7]


            elif len(list(set(input_list).intersection(set(["length", "lengths"])))) > 0:
                return XAI_User_Intents[8]



            elif len(list(set(input_list).intersection(set(["model","models"])))) > 0:
                return XAI_User_Intents[1]


            elif len(list(set(input_list).intersection(set(["quality","style","scores", "score"])))) > 0:
                return XAI_User_Intents[2]


            elif len(list(set(input_list).intersection(set(["label","labels", "aspect", "aspects", "structure", "structures"])))) > 0:
                return XAI_User_Intents[3]

            else:
                return "none"


        ###### Using an Intent Classifier ######
        if self.nlu_algorithm == "scibert_classifier":
            user_intent = self.intent_classification(input)
            return user_intent

        



        


# ###### For Debug ######
# def main():
#     # writingInput1 = ["What would the system predict if this instance changes to ...?", "Commercial speech applications are reducing human transcription of customer data."]
#     # writingInput2 = "Commercial speech applications are reducing human transcription of customer data."
#     writingInput1 = "How to change one label to another?"


#     root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
#     with open(os.path.join(root_dir, "xai_models/configs/xai_model_configs.json" ), 'r') as fp:
#         xai_model_configs = json.load(fp)

#     nlu = XAI_NLU_Module(xai_model_configs, nlu_algorithm = "scibert_classifier")

#     nlu(writingInput1)

# if __name__ == "__main__":
#     main()




#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code supports the web server of the ConvXAI system.
# Copyright (c) Hua Shen 2022.
#

import os
import sys
import json
import torch
import logging
import difflib
from .explainers import *
from .nlu import XAI_User_Intents
from convxai.writing_models.models import *


logger = logging.getLogger("my-logger")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT)
logger.setLevel(logging.INFO)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Model_Explainer(object):

    def __init__(self, model):
        super(Model_Explainer, self).__init__()
        logging.info("\nLoading writing models to be explained......")
        self.diversity_model  = model
        self.global_explanations_data = global_explanations_data
        

    def generate_explanation(self, user_intent_detection, writingInput, predictLabel, conference, **kwargs):
        """writingInput: List[str]. A list of all input text.
                writingInput = ' '.join(writingInput)
        """
        conference = str(conference).strip().upper()

        ############ Global Explanations ############
        if user_intent_detection == XAI_User_Intents[0]:
            explanations = self.explain_meta_data(conference)

        elif user_intent_detection == XAI_User_Intents[1]:
            explanations = self.explain_meta_model()

        elif user_intent_detection == XAI_User_Intents[2]:
            explanations = self.explain_quality_score(conference)

        elif user_intent_detection == XAI_User_Intents[3]:
            explanations = self.explain_aspect_distribution(conference)

        elif user_intent_detection == XAI_User_Intents[8]:
            explanations = self.explain_sentence_length(conference)


        ############ Local Explanations ############
        elif user_intent_detection == XAI_User_Intents[4]:
            explanations = self.explain_confidence(writingInput, **kwargs)

        elif user_intent_detection == XAI_User_Intents[5]:
            explanations = self.explain_example(writingInput, predictLabel, conference, **kwargs)
        
        elif user_intent_detection == XAI_User_Intents[6]:
            explanations = self.explain_attribution(writingInput, predictLabel, **kwargs)

        elif user_intent_detection == XAI_User_Intents[7]:
            explanations = self.explain_counterfactual(writingInput, predictLabel, **kwargs)


        else:
            explanations = "We currently can't answer this quesiton. Would you like to ask another questions?"
            
        return explanations



    ############ Global Explanations ############
    def explain_meta_data(self, conference):
        """Global Explanations."""
        explanation_dict = {
            "conference": conference,
            "global_explanations_data": self.global_explanations_data
        }
        return explanation_dict


    def explain_meta_model(self):
        """Global Explanations."""
        explanation_dict = {
            "explanation": None
        }
        return explanation_dict


    def explain_quality_score(self, conference):
        """Global Explanations."""
        explanation_dict = {
            "conference": conference,
            "global_explanations_data": self.global_explanations_data
        }
        return explanation_dict


    def explain_aspect_distribution(self, conference):
        """Global Explanations
        """
        explanation_dict = {
            "conference": conference,
            "global_explanations_data": self.global_explanations_data
        }
        return explanation_dict


    def explain_sentence_length(self, conference):
        """The [20th, 40th, 50th, 60th, 80th] percentiles of the sentence lengths in the conference are words. """
        explanation_dict = {
            "conference": conference,
            "global_explanations_data": self.global_explanations_data
        }
        return explanation_dict


    ############ Local Explanations ############
    def explain_confidence(self, input, **kwargs):
        """Local Explanations: XAI Algorithm: provide the output confidence of the writing support models.
        """
        ######### Explaining #########
        predict, probability = self.diversity_model.generate_confidence(input)     # predict: [4];  probability [0.848671]
        
        ######### NLG #########
        label = diversity_model_label_mapping[predict[0]]
        
        explanation_dict = {
            "input": input,
            "label": label,
            "probability": probability
        }
        return explanation_dict




    def explain_example(self, input, predictLabel, conference, **kwargs):
        """XAI Algorithm - Confidence: 
            Paper: An Empirical Comparison of Instance Attribution Methods for NLP (https://aclanthology.org/2021.naacl-main.75.pdf)
        """
        ######### User Input Variable #########
        top_k = kwargs["attributes"]["top_k"] if kwargs and kwargs["attributes"]["top_k"] is not None else 3

        ######### Explaining #########
        self.example_explainer = ExampleExplainer(conference)
        embeddings = self.diversity_model.generate_embeddings(input)

        if kwargs and kwargs["attributes"]["aspect"] is not None:
            label_idx = label_mapping[kwargs["attributes"]["aspect"]]
        else:
            # label = predictLabel
            label_idx = label_mapping[predictLabel]

        filter_index = np.where(self.example_explainer.diversity_aspect_list_tmp == label_idx)[0]
        self.example_explainer.diversity_x_train_embeddings_tmp = np.array(self.example_explainer.diversity_x_train_embeddings_tmp)[filter_index]
        similarity_scores = np.matmul(embeddings, np.transpose(self.example_explainer.diversity_x_train_embeddings_tmp, (1,0)))[0]    ###### train_emb = torch.Size(137171, 768)  

        if kwargs and kwargs["attributes"]["aspect"] is not None:
            final_top_index = filter_index[np.argsort(similarity_scores)[::-1][:top_k]]
        else:
            final_top_index =  np.argsort(similarity_scores)[::-1][:top_k]

        top_text = np.array(self.example_explainer.diversity_x_train_text_tmp)
        # top_title = np.array(self.example_explainer.diversity_x_train_title_tmp)
        top_link = np.array(self.example_explainer.diversity_x_train_link_tmp)
        # top_aspect = np.array(self.example_explainer.diversity_aspect_list_tmp)

        explanation_dict = {
            "top_k": top_k,
            "input": input,
            "conference": conference,
            "label": label_idx,
            "final_top_index": final_top_index,
            "top_link": top_link,
            "top_text": top_text
        }
        return explanation_dict


    def explain_attribution(self, input, predictLabel, **kwargs):
        """XAI Algorithm - Attribution: 
            Implementation Reference: https://stackoverflow.com/questions/67142267/gradient-based-saliency-of-input-words-in-a-pytorch-model-from-transformers-libr
        """
        top_k = kwargs["attributes"]["top_k"] if kwargs and kwargs["attributes"]["top_k"] is not None else 3
        
        if kwargs and kwargs["attributes"]["aspect"] is not None:
            label_idx = label_mapping[kwargs["attributes"]["aspect"]]
        else:
            label_idx = label_mapping[predictLabel]

        self.attribution_explainer = AttributionExplainer(self.diversity_model)
        all_predic_toks, ordered_predic_tok_indices = self.attribution_explainer.get_sorted_important_tokens(input, label_idx)
        important_indices = ordered_predic_tok_indices[:top_k]
        
        explanation_dict = {
            "top_k": top_k,
            "all_predic_toks": all_predic_toks,
            "important_indices": important_indices
        }
        return explanation_dict



    def explain_counterfactual(self, input, predictLabel=None, **kwargs):
        """XAI Algorithm #6: MICE
        Reference paper: Explaining NLP Models via Minimal Contrastive Editing (MICE)
        """
        y_pred_prob = self.diversity_model.get_probability(input)
        default_contrast_label_idx_input = np.argsort(y_pred_prob[0][0])[-2]
        contrast_label_idx_input = default_contrast_label_idx_input
        self.counterfactual_explainer = CounterfactualExplainer()
        output = self.counterfactual_explainer.generate_counterfactual(input, contrast_label_idx_input)
        if len(output) > 0:
            d = difflib.Differ()
            original = output['original_input'].replace(".", " ")
            edited = output['counterfactual_input'].replace(".", " ")
            diff = list(d.compare(original.split(), edited.split()))
            counterfactual_output = ""
            for d in diff:
                if d[:2] == "+ ":
                    counterfactual_output += f"<b><span style='font-weight: bold; background-color: #F9B261; border-radius: 5px;'>{d[2:]}</span></b>"
                    counterfactual_output += " "
                if d[:2] == "  ":
                    counterfactual_output += d[2:]
                    counterfactual_output += " "
            explanation_dict = {
                "counterfactual_exists": True,
                "output": output,
                "counterfactual_output": counterfactual_output
            }
        else:
            explanation_dict = {
                "counterfactual_exists": False,
                "output": output,
                "counterfactual_output": None
            }
        return explanation_dict






    # def explain_counterfactual(self, input, predictLabel=None, **kwargs):
    #     """XAI Algorithm #6: MICE
    #     Reference paper: Explaining NLP Models via Minimal Contrastive Editing (MICE)
    #     """
    #     contrast_label_idx_input = label_mapping[kwargs["attributes"]["contrast_label"]] if kwargs and kwargs["attributes"]["contrast_label"] is not None else -2
    #     self.counterfactual_explainer = CounterfactualExplainer()
    #     output = self.counterfactual_explainer.generate_counterfactual(input, contrast_label_idx_input)
    #     if len(output) > 0:
    #         d = difflib.Differ()
    #         original = output['original_input'].replace(".", " ")
    #         edited = output['counterfactual_input'].replace(".", " ")
    #         diff = list(d.compare(original.split(), edited.split()))
    #         counterfactual_output = ""
    #         for d in diff:
    #             if d[:2] == "+ ":
    #                 counterfactual_output += f"<b><span style='font-weight: bold; background-color: #F9B261; border-radius: 5px;'>{d[2:]}</span></b>"
    #                 counterfactual_output += " "
    #             if d[:2] == "  ":
    #                 counterfactual_output += d[2:]
    #                 counterfactual_output += " "
    #         explanation_dict = {
    #             "counterfactual_exists": True,
    #             "output": output,
    #             "counterfactual_output": counterfactual_output
    #         }
    #     else:
    #         explanation_dict = {
    #             "counterfactual_exists": False,
    #             "output": output,
    #             "counterfactual_output": None
    #         }
    #     return explanation_dict




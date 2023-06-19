#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code supports the web server of the ConvXAI system.
# Copyright (c) Hua Shen 2023.
#


import torch
import logging

from .nlu import XAI_User_Intent_Map

from .ai_models import (
    DiversityModel,
    QualityModel
)

from .ai_explainers import (
    global_explanations_data,
    explain_meta_data,
    explain_meta_model,
    explain_quality_score,
    explain_aspect_distribution,
    explain_sentence_length,
    explain_confidence,
    explain_example,
    explain_attribution,
    explain_counterfactual,
    ChatGPT
)

device = torch.device('cpu')


class AI_Explainers(object):
    """The AI_Explainers unifies a set of XAI algorithms, which uses the unified input output variables for diverse XAI algorithms.
    """

    def __init__(self,
                 configs: dict,
                 dialog_turns: int = 5):
        super(AI_Explainers, self).__init__()
        logging.info("\nLoading writing models to be explained......")

        # Load models
        self.diversity_model = DiversityModel(
            saved_model_dir=configs["scientific_writing"]["diversity_model_dir"])
        self.quality_model = QualityModel(
            saved_model_dir=configs["scientific_writing"]["quality_model_dir"])

        # Load ChatGPT and Dialog Tacker
        self.init_conversations = []
        self.explainer = ChatGPT()
        self.N = dialog_turns

    def explain(self,
                xai_user_intent: str,
                user_xai_question: str,
                ai_input_instance: str,
                ai_predict_output: str,
                **kwargs
                ):
        """The unified function of generating explanations.
        Args:
            xai_user_intent: the detected xai user intent.
            user_xai_question: the original input of xai user question.
            ai_input_instance: the ai_input_instance selected by the user from the frontend user interface.
            ai_predict_output: the prediction output of the model on the selected instance.

        Returns:
            The free-text conversational response, which will be sent to user.
        """


        ############ Global Explanations ############
        # Global explanations defines the meta information that describes the dataset, model, system and so forth in general. 
        # You can concretely defines numerous 'global XAI intent and approach' to describe each aspect of the system (e.g., dataset, model, biases, etc.).
        # In this system, we pre-define the output for each global explanation method.
        # 

        if xai_user_intent == XAI_User_Intent_Map[1]:
            explanations = explain_meta_data(
                kwargs['conference'], global_explanations_data)

        elif xai_user_intent == XAI_User_Intent_Map[3]:
            explanations = explain_meta_model()

        elif xai_user_intent == XAI_User_Intent_Map[7]:
            explanations = explain_quality_score(
                kwargs['conference'], global_explanations_data)

        elif xai_user_intent == XAI_User_Intent_Map[8]:
            explanations = explain_aspect_distribution(
                kwargs['conference'], global_explanations_data)

        elif xai_user_intent == XAI_User_Intent_Map[9]:
            explanations = explain_sentence_length(
                kwargs['conference'], global_explanations_data)



        # ############ Local Explanations ############
        # Local explanations explain the model's prediction for each instance input.
        # We include four common local explanation in this system, including confidence_score, example-based, attribution-based, counterfactuals explanations.

        elif xai_user_intent == XAI_User_Intent_Map[5]:
            explanations = explain_confidence(
                self.diversity_model, ai_input_instance, **kwargs)

        elif xai_user_intent == XAI_User_Intent_Map[6]:
            explanations = explain_example(
                self.diversity_model, ai_input_instance, ai_predict_output, kwargs['conference'], **kwargs)

        elif xai_user_intent == XAI_User_Intent_Map[2]:
            explanations = explain_attribution(
                self.diversity_model, ai_input_instance, ai_predict_output, **kwargs)

        elif xai_user_intent == XAI_User_Intent_Map[0]:
            explanations = explain_counterfactual(
                self.diversity_model, ai_input_instance, ai_predict_output, **kwargs)

        else:
            print("******* ChatGPT is Responding **********")

            input_text = f"The provided text to review is: {ai_input_instance}"
            input_text += f" The user is asking the question: {user_xai_question}."

            self.init_conversations.append(
                {"role": "user", "content": input_text}
            )

            responses = self.explainer.generate(self.init_conversations)
            explanations = responses["choices"][0]["message"]["content"]

        #### Add bot reply to history ###
        self.init_conversations.append(
            {"role": "assistant", "content": explanations}
        )

        #### Keep latest N turns ###
        self.init_conversations = self.init_conversations[:-self.N]

        return explanations

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code supports the web server of the ConvXAI system.
# Copyright (c) Hua Shen 2023.
#

import os
import json
import torch
import numpy as np

from typing import Dict, List
from collections import defaultdict
from parlai.core.agents import register_agent, Agent

from convxai.utils import *
from .pipeline import XAI_NLU_Module, AI_Explainers, XAI_NLG_Module
from .pipeline.xai_tutorials import AICommenter
from .pipeline.ai_explainers import global_explanations_data


device = torch.device('cpu')


@register_agent("xai")
class XaiAgent(Agent):
    """[Receiving and Sending messages]
    The XAIAgent class to define the interaction of receiving (i.e., observe function) and sending (i.e., act function) messages to the users. 
    1) Parse user XAI inputs;g
    2) Collect AI Predictions;
    3) Send to XAI explainer for explanations.
    """

    @classmethod
    def add_cmdline_args(cls, parser, partial_opt):
        parser.add_argument('--name', type=str,
                            default='Alice', help="The agent's name.")
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'XaiAgent'
        self.name = opt['name']
        self.xai_explainer = ConvxaiCoordinator()

    def _parse_user_xai_inputs(self, input: Dict[str, List]) -> Dict[str, List]:
        """This function collects all the AI predictions needed for explanations.
        """
        user_inputs = {
            "explainInput": json.loads(self.input["text"])["explainInput"],
            "writingInput": json.loads(self.input["text"])["writingInput"],
            "writingIndex": json.loads(self.input["text"])["writingIndex"],
            "inputTexts": self.input["predict_results"]["diversity_model_outputs"]["inputs"]
        }
        return user_inputs

    def _parse_ai_predictions(self, input: Dict[str, List]) -> Dict[str, List]:
        """This function collects all the AI predictions needed for explanations.
        """
        outputs_predict = input["predict_results"]['diversity_model_outputs']['outputs_predict']
        outputs_predict_list = [int(v) for v in outputs_predict.split(',')]

        outputs_probability = input["predict_results"]['diversity_model_outputs']['outputs_probability']
        outputs_probability_list = [float(v)
                                    for v in outputs_probability.split(',')]

        outputs_perplexity = input["predict_results"]['quality_model_outputs']['outputs_perplexity']
        outputs_perplexity_list = [float(v)
                                   for v in outputs_perplexity.split(',')]
        ai_predictions = {
            "outputs_predict_list": outputs_predict_list,
            "outputs_probability_list": outputs_probability_list,
            "outputs_perplexity_list": outputs_perplexity_list
        }
        return ai_predictions

    def observe(self, observation):
        """Receive user XAI inputs and AI predictions to be used for generating explanations.
        """
        self.input = observation if observation else "Invalid input. Please ask your question again."
        self.user_inputs = self._parse_user_xai_inputs(self.input)
        self.ai_predict_outputs = self._parse_ai_predictions(self.input)

    def act(self):
        """Generate AI explanations and send back to users.
        """
        response_text, response_indicator = self.xai_explainer.explain(
            self.user_inputs, self.ai_predict_outputs)
        response = {
            "text": response_text,
            "writingIndex": response_indicator,
            "responseIndicator": [response_indicator],
        }
        return response


class ConvxaiCoordinator(object):
    """ConvxaiCoordinator class coordinates the whole conversation, which includes the functions of:
    1) XAI_NLU_Module: parses the user XAI question into a XAI intent that corresponds to a specific XAI algorithm.
    2) AI_Explainers: generates the intented AI explanation that user needs.
    3) XAI_NLG_Module: converts the output of AI_Explainers into natural language text if needed.
    4) Multi_turn_dialog_tracker: tracks the key variables to decide if the turn is for human customization or drill down query.
    """

    def __init__(self) -> None:

        # # ****** Load config file ****** #
        self.configs = parse_system_config_file()
        intent_model_path = self.configs["conversational_xai"]["intent_model"]

        # ****** Set up Convxai modules ****** #
        self.nlu = XAI_NLU_Module(intent_model_path)
        self.explainer = AI_Explainers(self.configs)
        self.nlg = XAI_NLG_Module()

        # ****** Set up convxai_global_status_track to track the multi-turn variables ****** #
        self.convxai_global_status_track = defaultdict()
        self.convxai_intent_round = 0
        # self.multi_turn_required_intent_list = ["example", "attribution", "counterfactual"]   # *** specify the XAI types that requires multi-turn tracks.

        # *** specify the XAI types that requires multi-turn tracks.
        self.multi_turn_required_intent_list = [
            "[similar examples]", "[important words]", "[counterfactual prediction]"]

        self.global_explanations_data = global_explanations_data




    def _check_multi_turn_input_variables(self, user_intent, user_input):
        """Check and handle if the turn is a follow-up for cusmtomization."""

        self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["user_intent"] = user_intent

        if user_intent in self.multi_turn_required_intent_list:

            if self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["binded"] is False:

                if user_intent == self.multi_turn_required_intent_list[0]:
                    if self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"] is None:
                        self.convxai_global_status_track[
                            f"intent_round_{self.convxai_intent_round}"]["binded"] = True
                        self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"] = {
                            "top_k": 3, "aspect": None, "keyword": None, "rank": None}
                        check_response = """Want to check more examples with other conditions?
                        <br><br>
                        You can specify:
                        <br> -<span class='text-danger font-weight-bold'>label:your_label</span>.  The selected examples are all predicted as this label.
                        <br> -<span class='text-danger font-weight-bold'>keyword:your_keyword</span>.  All selected exampls contain this keyword.
                        <br> -<span class='text-danger font-weight-bold'>rank:your_rank_method</span>. The selected examples are also ranked by quality score ('quality_score'), or longer length ('long'), or shorter length ('short').
                        <br> -<span class='text-danger font-weight-bold'>count:your_top_k</span>. How many examples to be shown.
                        <br><br> Please specify ONE or MORE of them by
                        <span class='font-weight-bold'>'label:your_label, keyword:your_keyword, rank:your_rank_method, count:your_top_k'</span>
                        <br>(e.g., <span class='text-danger font-weight-bold'>'count:8'</span>, or <span class='text-danger font-weight-bold'>'label:background, keyword:time, rank:quality_score, count:6'</span>).
                        """
                        return check_response

                if user_intent == self.multi_turn_required_intent_list[1]:
                    if self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"] is None:
                        self.convxai_global_status_track[
                            f"intent_round_{self.convxai_intent_round}"]["binded"] = True
                        self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"] = {
                            "top_k": 3, "aspect": None}
                        check_response = "Want to highlight <span class='text-danger font-weight-bold'>more important words for another label</span>? Please type <span class='text-danger font-weight-bold'>'word number + label'</span> (e.g., '6 + method'), <br><br> Or just say '<span class='text-danger font-weight-bold'>No</span>' if you don't need more examples : )."
                        return check_response

                if user_intent == self.multi_turn_required_intent_list[2]:
                    if self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"] is None:
                        self.convxai_global_status_track[
                            f"intent_round_{self.convxai_intent_round}"]["binded"] = True
                        self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"] = {
                            "contrast_label": None}
                        check_response = "Would you like to <span class='text-danger font-weight-bold'>set another contrastive label</span> to change to? Please type the label from <span class='text-danger font-weight-bold'>'background', 'method', 'purpose', 'finding', 'others'</span>, or reply 'No' if you won't need."
                        return check_response

            elif self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["binded"] is True:

                user_input = str(user_input).lower().strip()
                user_input = user_input.replace(" ", "")

                if user_intent == self.multi_turn_required_intent_list[0]:
                    # Parse the user input variables in text.
                    # example: "rank:quality, label:background, keyword:time, count:6"
                    key_list = user_input.split(":")
                    if len(key_list) == 0:
                        if user_input in ["no", "NO", "No"]:
                            response = [
                                "no", "OK, great! It seems you are satisfied with current results. Please feel free to ask more questions : )"]
                            return response
                        else:
                            check_response = """Your input is not in correct format, 
                                <br><br> Please specify ONE or MORE of them by
                                <span class='font-weight-bold'>'label:your_label, keyword:your_keyword, rank:your_rank_method, count:your_top_k'</span>
                                <br>(e.g., <span class='text-danger font-weight-bold'>'count:8'</span>, or <span class='text-danger font-weight-bold'>'label:background, keyword:time, rank:quality_score, count:6'</span>)."""
                            return check_response
                    else:
                        # Add key values into variable dictionary.
                        user_input_dict = {}
                        all_tokens = []
                        for k in range(len(key_list)):
                            all_tokens.extend(
                                key_list[k].replace(",", " ").split())

                        for t in range(len(all_tokens)):
                            if all_tokens[t] in ['rank', 'count', 'label', 'keyword']:
                                value = all_tokens[t+1]
                                user_input_dict[all_tokens[t]] = value
                        try:
                            if 'count' in user_input_dict.keys():
                                self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"]["top_k"] = int(
                                    user_input_dict['count'])
                            if 'label' in user_input_dict.keys():
                                self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"][
                                    "attributes"]["aspect"] = user_input_dict['label']
                            if 'keyword' in user_input_dict.keys():
                                self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"][
                                    "attributes"]["keyword"] = user_input_dict['keyword']
                            if 'rank' in user_input_dict.keys():
                                self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"][
                                    "attributes"]["rank"] = user_input_dict['rank']
                            self.convxai_global_status_track[
                                f"intent_round_{self.convxai_intent_round}"]["binded"] = False
                            return "continue"
                        except:
                            check_response = """Your input is not in correct format, 
                                <br><br> Please specify ONE or MORE of them by
                                <span class='font-weight-bold'>'label:your_label, keyword:your_keyword, rank:your_rank_method, count:your_top_k'</span>
                                <br>(e.g., <span class='text-danger font-weight-bold'>'count:8'</span>, or <span class='text-danger font-weight-bold'>'label:background, keyword:time, rank:quality_score, count:6'</span>)."""
                            return check_response

                if user_intent == self.multi_turn_required_intent_list[1]:
                    if "+" in user_input:
                        user_input, aspect = user_input.split("+")
                        if not user_input.isdigit() or aspect not in ["background", "purpose", "method", "finding", "other"]:
                            if user_input in ["no", "NO", "No"]:
                                response = [
                                    "no", "OK, great! It seems you are satisfied with current results. Please feel free to ask more questions : )"]
                                return response
                            else:
                                check_response = "Your input is not in correct format, please type only one number (e.g., 6) or 'No' as your input."
                                return check_response
                        else:
                            self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"]["top_k"] = int(
                                user_input)
                            self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"][
                                "attributes"]["aspect"] = aspect
                            self.convxai_global_status_track[
                                f"intent_round_{self.convxai_intent_round}"]["binded"] = False
                            return "continue"

                if user_intent == self.multi_turn_required_intent_list[2]:
                    if user_input not in ["background", "purpose", "method", "finding", "other"]:
                        if user_input in ["no", "NO", "No"]:
                            response = [
                                "no", "OK, great! It seems you are satisfied with current results. Please feel free to ask more questions : )"]
                            return response
                        else:
                            check_response = "Your input is not a correct label, please type one from <strong>['background', 'purpose', 'method', 'finding', 'other']</strong> below."
                            return check_response
                    else:
                        self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"] = {
                            "contrast_label": user_input}
                        self.convxai_global_status_track[
                            f"intent_round_{self.convxai_intent_round}"]["binded"] = False
                        return None

        elif user_intent not in self.multi_turn_required_intent_list:
            return None

    def explain(self,
                user_inputs: dict,
                ai_predict_outputs: list,
                response_indicator: int = 1,
                **kwargs
                ):

        # string
        user_xai_question = user_inputs['explainInput']
        # list of ai input sentences
        ai_input_instances = user_inputs['writingInput']
        # list of ai input index strings
        ai_input_indexes = user_inputs['writingIndex']
        # list of ai aspect strings
        ai_input_texts = user_inputs['inputTexts']

        ####################################################################
        # ****** [Generating AI Explanations] ****** #
        ####################################################################
        ai_predict_labels = [
            ai_predict_outputs["outputs_predict_list"][int(k)] for k in ai_input_indexes]

        ####################################################################
        # ****** [Init-Conversation]: welcome sentences + Generate AI Comments ****** #
        ####################################################################
        if user_xai_question in ["ACL", "CHI", "ICLR"]:
            self.convxai_global_status_track["conference"] = user_xai_question
            self.ai_commenter = AICommenter(
                user_xai_question, ai_input_instances, ai_predict_outputs, ai_input_texts)
            response = self.ai_commenter.ai_comment_generation()
            response_init_xai = "Would you need some <span class='text-danger font-weight-bold'>hints</span> about how to <span class='text-danger font-weight-bold'>use XAI</span> to <span class='text-danger font-weight-bold'>improve sentence writing</span>?"
            return [response, response_init_xai], 2

        user_xai_question = user_xai_question.strip().lower()



        ####################################################################
        # ****** [User Intent Classification] ****** #
        ####################################################################
        user_xai_question = user_xai_question.strip().lower()

        if f"intent_round_{self.convxai_intent_round}" not in self.convxai_global_status_track.keys():
            ###### """Initiate the 'conversation_intent_round_k' dictionary """ ######
            self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"] = {
                "user_intent": None,
                "binded": False,
                "attributes": None
            }
        
        print("======>>>>>>!!!!!self.nlu(user_xai_question)[0]", self.nlu(user_xai_question)[0])

        if self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["binded"] is True:
            # xai_user_intent = self.convxai_global_status_track[
            #     f"intent_round_{self.convxai_intent_round}"]['user_intent']
            user_intent_check = self.nlu(user_xai_question)[0]
            if user_intent_check != "[other]":
                xai_user_intent = user_intent_check
                if user_intent_check != self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]['user_intent']:
                    self.convxai_intent_round += 1
                    self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"] = {
                        "user_intent": None,
                        "binded": False,
                        "attributes": None
                    }
                    self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["binded"] = False
            else:
                xai_user_intent = self.convxai_global_status_track[
                    f"intent_round_{self.convxai_intent_round}"]['user_intent']
        else:
            xai_user_intent = self.nlu(user_xai_question)[0]


        print(f"===> Detected User Intent = {xai_user_intent}")

        ###### Generating Global AI Explanations ######
        # if user_intent in ["ai-comment-global", "meta-data", "meta-model", "quality-score", "aspect-distribution", "sentence-length"]:
        if xai_user_intent in ["[data statistics]", "[model description]", "[other]", "[quality score]", "[label distribution]", "[sentence length]"]:

            # if user_intent == "ai-comment-global":
            #     response = "<strong>-Basic Information and Statistics</strong> of the data and model for generating your reviews above:<br>"
            #     self.convxai_intent_round += 1
            #     return [response], 3

            explanations = self.explainer.explain(
                xai_user_intent,
                user_xai_question,
                "",
                "",
                conference=self.convxai_global_status_track["conference"],
                **self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"])
            response = self.nlg(explanations)
            self.convxai_intent_round += 1
            return [response], response_indicator

        ###### Generating Local AI Explanations ######
        # elif user_intent in ["ai-comment-instance", "confidence", "example", "attribution", "counterfactual"]:
        elif xai_user_intent in ["[prediction confidence]", "[similar examples]", "[important words]", "[counterfactual prediction]"]:

            ###### Ensure the user select one sentence for explanations ######
            if len(ai_input_instances) == 0:
                response = "You are asking for <strong>instance-wise explanations</strong>, and <strong>please select (double click) one sentence</strong> to be explained."
                return [response], response_indicator

            # ###### Explaining the AI comments. ######
            # if user_intent == "ai-comment-instance":
            #     response = self.ai_commenter.explaining_ai_comment_instance(list(set(ai_input_indexes)), self.convxai_global_status_track)
            #     self.convxai_intent_round += 1
            #     return response, 4

            ###### Collect the variables to generate explanations. ######
            check_response = self._check_multi_turn_input_variables(
                xai_user_intent, user_xai_question)

            responses = []
            for k in range(len(ai_input_instances)):
                explanations = self.explainer.explain(
                    xai_user_intent,
                    user_xai_question,
                    ai_input_instances[k],
                    ai_predict_labels[k],
                    conference=self.convxai_global_status_track["conference"],
                    **self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]
                )
                response = self.nlg(explanations)
                responses.append(response)

            if check_response is not None:
                if check_response == "continue":
                    self.convxai_intent_round += 1
                    return responses, response_indicator
                elif len(check_response) == 2:
                    return [check_response[1]], response_indicator
                else:
                    return [responses, check_response], 6
            else:
                self.convxai_intent_round += 1
                return responses, response_indicator

        else:
            explanations = "We currently can't answer this quesiton. Would you like to ask another questions?"
            response = self.nlg(explanations)
            self.convxai_intent_round += 1
            return [response], response_indicator

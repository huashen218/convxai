#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code supports the unified XAI API in ConvXAI system.
# 
# Copyright (c) Hua Shen 2023.
#

import os
import json
import torch
from collections import defaultdict
from IPython.core.display import display, HTML

from .nlu import XAI_NLU_Module
from .explainer import AI_Explainers
from .nlg import XAI_NLG_Module
from .xai_tutorials import AICommenter
from .ai_explainers import global_explanations_data
from .ai_models import *
from convxai.utils import *



# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets


device = torch.device('cpu')



class ConvXAI(object):
    """ConvXAI class coordinates the whole conversation, which includes the functions of:
    1) XAI_NLU_Module: parses the user XAI question into a XAI intent that corresponds to a specific XAI algorithm.
    2) AI_Explainers: generates the intented AI explanation that user needs.
    3) XAI_NLG_Module: converts the output of AI_Explainers into natural language text if needed.
    4) Convxai_Global_Status_Track: tracks the key variables to decide if the turn is for human customization or drill down query.
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
                user_xai_question: str, 
                ai_input: str, 
                ai_predict_output:str, 
                conference: str = "CHI",
                multi_turn: bool = False,
                visualize: bool = False):
        """AI explanation function.
        """

        self.convxai_global_status_track["conference"] = conference
        user_xai_question = user_xai_question.strip().lower()
        ai_predict_output = label_mapping[ai_predict_output]

        if f"intent_round_{self.convxai_intent_round}" not in self.convxai_global_status_track.keys():
            ###### """Initiate the 'conversation_intent_round_k' dictionary """ ######
            self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"] = {
                "user_intent": None,
                "binded": False,
                "attributes": None
            }
        
        if self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["binded"] is True:
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


        ###### Generating Global AI Explanations ######
        if xai_user_intent in ["[data statistics]", "[model description]", "[other]", "[quality score]", "[label distribution]", "[sentence length]"]:

            explanations = self.explainer.explain(
                xai_user_intent,
                user_xai_question,
                ai_input,
                ai_predict_output,
                conference=self.convxai_global_status_track["conference"],
                **self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]
                )
            response = [self.nlg(explanations)]
            self.convxai_intent_round += 1
            if visualize:
                self._visualize_single_turn_dialog(user_xai_question, response)
            return response


        ###### Generating Local AI Explanations ######
        elif xai_user_intent in ["[prediction confidence]", "[similar examples]", "[important words]", "[counterfactual prediction]"]:

            ###### Collect the variables to generate explanations. ######
            check_response = self._check_multi_turn_input_variables(
                xai_user_intent, user_xai_question)


            explanations = self.explainer.explain(
                xai_user_intent,
                user_xai_question,
                ai_input,
                ai_predict_output,
                conference=self.convxai_global_status_track["conference"],
                **self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]
            )
            response = [self.nlg(explanations)]


            if check_response is not None:
                if check_response == "continue":
                    self.convxai_intent_round += 1
                elif len(check_response) == 2:
                    response = [check_response[1]]
                else:
                    response.append(check_response)
                if visualize:
                    self._visualize_single_turn_dialog(user_xai_question, response)
                return response
            else:
                self.convxai_intent_round += 1
                if visualize:
                    self._visualize_single_turn_dialog(user_xai_question, response)
                return response

        else:
            explanations = "We currently can't answer this quesiton. Would you like to ask another questions?"
            response = [self.nlg(explanations)]
            self.convxai_intent_round += 1
            if visualize:
                self._visualize_single_turn_dialog(user_xai_question, response)
            return response



    def _visualize_single_turn_dialog(self, user_input, responses):
        message_html = """<div style='float: left; width: 50%; padding: 4px 10px; border-radius: 6px 6px 1px 6px; background: #347AB7; color: white;'>
            {text}
        </div>
        <div style='float: left;'>
            :👩🏻‍💻
        </div>
        """
        reply_html = """
        <div style='float: left;'>
            👨🏼‍🏫:
        </div>
        <div style='float: left; width: 50%; padding: 4px 10px; border-radius: 6px 6px 6px 1px; background: #B1CFE7;'>
            {text}
        </div>
        """
        display(HTML(message_html.format(text=user_input)))
        for response in responses:
            display(HTML(reply_html.format(text=response)))



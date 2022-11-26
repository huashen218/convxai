#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code supports the web server of the ConvXAI system.
# Copyright (c) Hua Shen 2022.
#

import os
import json
import torch
import joblib
import logging
import numpy as np
from dtaidistance import dtw
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
from collections import defaultdict
from parlai.core.agents import register_agent, Agent

from .modules import *
from convxai.utils import *
from convxai.writing_models.models import *


logger = logging.getLogger("my-logger")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT)
logger.setLevel(logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@register_agent("xai")
class XaiAgent(Agent):
    """[Receiving and Sending messages]
    The XAIAgent class to define the interaction of receiving (i.e., observe function) and sending (i.e., act function) messages to the users. 
    1) Parse user XAI inputs;
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
        self.xai_explainer = XAIExplainer()

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
        self.predictOutputs = self._parse_ai_predictions(self.input)

    def act(self):
        """Generate AI explanations and send back to users.
        """
        response_text, response_indicator = self.xai_explainer.explain(self.user_inputs['explainInput'], self.user_inputs['writingInput'], self.predictOutputs, self.user_inputs['inputTexts'],self.user_inputs['writingIndex'])

        response = {
            "text": response_text,
            "writingIndex": response_indicator,
            "responseIndicator": [response_indicator],
        }
        return response

class AICommenter(object):
    """Based on AI predictions, AICommenter generate high-level AI comments that are more understandable and useful for users in practice.
    """
    def __init__(self, conference, writingInput, predictOutputs, inputTexts):
        self.conference = conference
        self.predictOutputs = predictOutputs
        self.writingInput = writingInput
        self.inputTexts = inputTexts
        self.review_summary = {}
        self.aspect_model = TfidfAspectModel(self.conference)
        self.revision_comment_template = {
            "shorter_length": "&nbsp;&nbsp;<p class='comments' id={id} class-id=shorter-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: The sentence is <strong>too short</strong>, the average length of the sentences predicted as <strong>'{label}'</strong> labels in {conference} conference is {ave_word} words. Please rewrite it into a longer one.</p><br>",
            "longer_length": "&nbsp;&nbsp;<p class='comments' id={id} class-id=longer-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: The sentence is <strong>too long</strong>, the average length of the sentences predicted as <strong>'{label}'</strong> labels in {conference} conference is {ave_word} words. Please rewrite it into a shorter one.</p><br>",
            "label_change": "&nbsp;&nbsp;<p class='comments' id={id} class-id=label-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: Based on the sentence <strong>labels' percentage and order</strong> in your abstract, it is suggested to write your <strong>{aspect_new}</strong> at this sentence, rather than describing <strong>{aspect_origin}</strong> here.</p><br>",
            "low_score": "&nbsp;&nbsp;<p class='comments' id={id} class-id=score-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: The writing style quality score of {sentence} is a bit <strong>lower</strong> than <strong>'{label}'</strong>-labeled sentences in the {conference} conference. This indicate the writing style might not match well with this conference.<br>"
        }
        self.global_explanations_data = global_explanations_data
            
    def _ai_comment_score_classifier(self, raw_score, score_benchmark):
        if raw_score >= score_benchmark[4]:
            score_label = 1
        if raw_score >= score_benchmark[3] and raw_score < score_benchmark[4]:
            score_label = 2
        if raw_score >= score_benchmark[1] and raw_score < score_benchmark[3]:
            score_label = 3
        if raw_score >= score_benchmark[0] and raw_score < score_benchmark[1]:
            score_label = 4
        if raw_score < score_benchmark[0]:
            score_label = 5
        return score_label

    def _ai_comment_analyzer(self):

        structure_score, quality_score = 5, 0

        ###### Benchmark Scores ######
        abstract_score_benchmark = self.global_explanations_data[self.conference]['abstract_score_range']
        sentence_score_benchmark = self.global_explanations_data[self.conference]['sentence_score_range']
        sentence_length_benchmark = self.global_explanations_data[self.conference]['sentence_length']

        ###### Quality Scores ######
        sentence_raw_score = self.predictOutputs["outputs_perplexity_list"]
        abstract_score = self._ai_comment_score_classifier(np.mean(sentence_raw_score), abstract_score_benchmark)

        ###### Aspects Labels & Patterns ######
        aspect_list = self.predictOutputs["outputs_predict_list"]  # aspect_feedback
        aspect_distribution_benchmark = self.aspect_model.predict(aspect_list)
        aspect_distribution_benchmark = aspect_distribution_benchmark.aspect_sequence

        length_revision = ""
        structure_revision = ""
        style_revision = ""

        # Aspects Labels Analysis ######: DTW algorithm: https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html#dtw-distance-measure-between-two-time-series;   https://towardsdatascience.com/an-illustrative-introduction-to-dynamic-time-warping-36aa98513b98
        distance, paths = dtw.warping_paths(
            aspect_list, aspect_distribution_benchmark)
        best_path = dtw.best_path(paths)
        change_label_dict = {}
        for path in best_path:
            if aspect_list[path[0]] != aspect_distribution_benchmark[path[1]]:
                change_label_dict[path[0]] = path[1]

        ###### Length and Score Analysis ######
        for n in range(len(sentence_raw_score)):
            sentence_score = self._ai_comment_score_classifier(
                sentence_raw_score[n], sentence_score_benchmark[diversity_model_label_mapping[aspect_list[n]]])
            sentence_length = self._ai_comment_score_classifier(len(counting_tokens(
                self.inputTexts[n])), sentence_length_benchmark[diversity_model_label_mapping[aspect_list[n]]])

            if n in change_label_dict.keys():
                k = change_label_dict[n]
                structure_revision += self.revision_comment_template["label_change"].format(
                    id=f"{n}", sentence=f"S{n+1}", aspect_origin=diversity_model_label_mapping[aspect_list[n]], aspect_new=diversity_model_label_mapping[aspect_distribution_benchmark[k]])
                self.review_summary.setdefault(n, []).append(
                    f"aspect-{diversity_model_label_mapping[aspect_list[n]]}-{diversity_model_label_mapping[aspect_distribution_benchmark[k]]}")

            if sentence_score < 2:
                style_revision += self.revision_comment_template["low_score"].format(
                    id=f"{n}", sentence=f"S{n+1}", label=diversity_model_label_mapping[aspect_list[n]], conference=self.conference)
                self.review_summary.setdefault(n, []).append(
                    f"quality-{sentence_raw_score[n]}")

            if sentence_length > 4:
                length_revision += self.revision_comment_template["shorter_length"].format(
                    id=f"{n}", sentence=f"S{n+1}", conference=self.conference, label=diversity_model_label_mapping[aspect_list[n]], ave_word=self.global_explanations_data[self.conference]["sentence_length"][diversity_model_label_mapping[aspect_list[n]]][2])
                self.review_summary.setdefault(n, []).append(
                    f"short-{len(counting_tokens(self.inputTexts[n]))}")

            if sentence_length < 2:
                length_revision += self.revision_comment_template["longer_length"].format(
                    id=f"{n}", sentence=f"S{n+1}", conference=self.conference, label=diversity_model_label_mapping[aspect_list[n]], ave_word=self.global_explanations_data[self.conference]["sentence_length"][diversity_model_label_mapping[aspect_list[n]]][2])
                self.review_summary.setdefault(n, []).append(
                    f"long-{len(counting_tokens(self.inputTexts[n]))}")

            quality_score += sentence_score

        self.review_summary["abstract_score_benchmark"] = self.global_explanations_data[self.conference]['abstract_score_range']
        self.review_summary["sentence_score_benchmark"] = self.global_explanations_data[
            self.conference]['sentence_score_range'][diversity_model_label_mapping[aspect_list[n]]]
        self.review_summary["sentence_length_benchmark"] = self.global_explanations_data[
            self.conference]['sentence_length'][diversity_model_label_mapping[aspect_list[n]]]
        self.review_summary["prediction_label"] = diversity_model_label_mapping[aspect_list[n]]
        aspect_keys = "".join(list(map(str, aspect_distribution_benchmark)))
        self.review_summary["aspect_distribution_benchmark"] = self.global_explanations_data[self.conference]["Aspect_Patterns_dict"][aspect_keys]

        if len(length_revision) == 0:
            length_revision = "&nbsp;&nbsp;<p><strong>-</strong>Your sentence lengths look good to me. Great job!</p>"

        if len(structure_revision) == 0:
            structure_revision = "&nbsp;&nbsp;<p><strong>-</strong>Your abstract structures look good to me. Great job!</p>"

        if len(style_revision) == 0:
            style_revision = "&nbsp;&nbsp;<p><strong>-</strong>Your writing styles look good to me. Great job!</p>"

        feedback_improvement = f"<br> <p style='color:#1B5AA2;font-weight:bold'> Structure Suggestions:</p> {structure_revision}" + \
                               f"<br> <p style='color:#1B5AA2;font-weight:bold'> Style Suggestions:</p> {style_revision} {length_revision}"

        structure_score = 5 - 0.5 * len(change_label_dict.keys())
        overall_score = (structure_score + quality_score /
                         len(sentence_raw_score)) / 2

        analysis_outputs = {
            "abstract_score": overall_score,
            "instance_results": feedback_improvement
        }

        return analysis_outputs

    def ai_comment_generation(self):
        """Generate the AI comments, which include:
        1) the a brief statistic of the selected conference;
        2) the generated writing score;
        3) how to improvement for specific sentences.
        """
        analysis_outputs = self._ai_comment_analyzer()
        comment_conference_intro = f"Nice! I'm comparing your submission with {self.global_explanations_data[self.conference]['paper_count']} {self.conference} paper abstracts."
        comment_overall_score = f"<br><br><p class='overall-score'> Your <strong>Overall Score</strong> of Structure and Style = <strong>{analysis_outputs['abstract_score']:0.0f}</strong> (out of 5).</p> "
        comment_improvements = "<br>" + analysis_outputs['instance_results'] if len(analysis_outputs['instance_results']) > 0 else "Your current writing looks good to me. Great job!"
        return comment_conference_intro + comment_overall_score + comment_improvements

    def _explaining_ai_comment_template(self, idx, review, convxai_global_status_track):

        review_type = review.split("-")[0]

        if review_type == "long":

            response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: we are comparing the <strong>length</strong> of this sentence with all the sentences' in {convxai_global_status_track["conference"]} dataset.
            Your sentence has <strong>{review.split("-")[1]} words</strong>, which is longer than 80% of the <strong>'{self.review_summary['prediction_label']}'</strong>-labeled sentences in the {convxai_global_status_track["conference"]} conference.
            <br><br>
            <strong>To improve</strong>, the sentence is suggested to have <strong>{self.review_summary['sentence_length_benchmark'][1]} to {self.review_summary['sentence_length_benchmark'][3]}</strong> words.
            <br><br>
            To see the more details of sentence length statistics, please click the question below:<br><br>
            """

        elif review_type == "short":
            response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: we are comparing the <strong>length</strong> of this sentence with all the sentences' in {convxai_global_status_track["conference"]} dataset.
            This sentence has <strong>{review.split("-")[1]} words</strong>, which is shorter than 80% of the <strong>'{self.review_summary['prediction_label']}'</strong>-labeled sentences in the {convxai_global_status_track["conference"]} conference.
            <br><br>
            <strong>To improve</strong>, the sentence is suggested to have <strong>{self.review_summary['sentence_length_benchmark'][1]} to {self.review_summary['sentence_length_benchmark'][3]}</strong> words.
            <br><br>
            To see the more details of sentence length statistics, please click the question below:<br><br>
            """

        elif review_type == "quality":
            response = f"""
            <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: the <strong>Writing Style Model</strong> can generate a quality score, indicating <strong>how well the sentence can match with the {convxai_global_status_track["conference"]} conference</strong>, with <strong>lower the better</strong> style quality.
            <br><br>
            This sentence gets <strong>{float(review.split("-")[1]):.2f}</strong> points, which is larger than <strong>80%</strong> of the <strong>'{self.review_summary['prediction_label']}'</strong>-labeled sentences in the {convxai_global_status_track["conference"]} conference. This indicates the sentence' writing style might not match well with published {convxai_global_status_track["conference"]} sentences. 
            <br><br>
            <strong>To improve</strong>, you can check {convxai_global_status_track["conference"]} similar sentences for reference. Please click the question below:<br><br>
            """

        elif review_type == "aspect":
            response = f"""
            <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: we summarized all the collected {convxai_global_status_track["conference"]} abstracts into <strong>five structural patterns</strong>, where we found your submission is closest to the pattern of <span class='text-danger font-weight-bold'>{self.review_summary["aspect_distribution_benchmark"]}</span>. By using <a class='post-link' href='https://en.wikipedia.org/wiki/Dynamic_time_warping' target='_blank'><strong>Dynamic Time Warping</strong></a> algorithm to analyze <strong> how to revise your submission to fit this style pattern</strong>,
            the result suggested to describe <strong>{review.split("-")[2]}</strong> aspect but not <strong>{review.split("-")[1]}</strong> in this sentence.
            <br><br>
            <strong>To improve</strong>, you can check the <strong> most important words </strong> resulting in the prediction and further check <strong> how to revise input into another label</strong> . See XAI questions below:<br><br>
            """

        return [review_type, response]

    def explaining_ai_comment_instance(self, writingIndex, convxai_global_status_track):
        response = []
        for idx in writingIndex:
            if int(idx) in self.review_summary.keys():
                for item in self.review_summary[int(idx)]:
                    response.extend(
                        self._explaining_ai_comment_template(int(idx), item, convxai_global_status_track))
        if len(response) == 0:
            response = [
                "none", "Your writing looks good to us! <br><br><strong>To improve</strong>, you can ask for explanation questions below:<br><br>"]
        return response




class XAIExplainer(object):
    """XAI generation class.
    XaiExplainer class defines how to generate the conversastional AI explanation for each classs.
    1) Maintain multi-turn AI conversations:
        - NLU Module;
        - XAI Explainer;
        - NLG Module;
    2) Generate AI Comments;
    3) Explain AI comment & predictions;
    """

    def __init__(self) -> None:

        # # ****** Load config file ****** #
        self.configs = parse_system_config_file()

        # ****** Set up Convxai modules ****** #
        self.nlu = XAI_NLU_Module(self.configs, nlu_algorithm="rule_based")
        self.explainer = Model_Explainer(self.configs)
        self.nlg = XAI_NLG_Module()

        # ****** Set up convxai_global_status_track ****** #
        self.convxai_global_status_track = defaultdict()
        self.convxai_intent_round = 0
        self.multi_turn_required_intent_list = [
            "example", "attribution", "counterfactual"]   # *** specify the XAI types that requires multi-turn tracks.
        self.global_explanations_data = global_explanations_data

    def _check_multi_turn_input_variables(self, user_intent, user_input):

        self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["user_intent"] = user_intent

        if user_intent in self.multi_turn_required_intent_list:

            if self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["binded"] is False:
                if user_intent == "example":
                    if self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"] is None:
                        self.convxai_global_status_track[
                            f"intent_round_{self.convxai_intent_round}"]["binded"] = True
                        self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"] = {
                            "top_k": 3, "aspect": None}
                        check_response = "Would you like to <span class='text-danger font-weight-bold'>see more or less examples</span>, and meanwhile conditioned on an aspect? If you need, please type the <span class='text-danger font-weight-bold'>word number + aspect (e.g., 6 + method)</span>, otherwise, please reply 'No'."
                        return check_response

                if user_intent == "attribution":
                    if self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"] is None:
                        self.convxai_global_status_track[
                            f"intent_round_{self.convxai_intent_round}"]["binded"] = True
                        self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"] = {
                            "top_k": 3}
                        check_response = "Would you like to <span class='text-danger font-weight-bold'>highlight more or less important words</span>? If you need, please type the word number (e.g., 6), otherwise, please reply 'No'."
                        return check_response

                if user_intent == "counterfactual":
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

                if user_intent == "example":
                    if "+" in user_input:
                        user_input, aspect = user_input.split("+")
                        if not user_input.isdigit() or aspect not in ["background", "purpose", "method", "finding", "other"]:
                            if user_input in ["no", "NO", "No"]:
                                response = [
                                    "no", "OK, great! It seems you are satisfied with current results. Please feel free to ask more questions : )"]
                                return response
                            else:
                                check_response = "Your input is not in correct format, please type the word number + aspect (e.g., 6 + method), otherwise, please reply 'No'."
                                return check_response
                        else:
                            self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"] = {
                                "top_k": int(user_input), "aspect": aspect}
                            self.convxai_global_status_track[
                                f"intent_round_{self.convxai_intent_round}"]["binded"] = False
                            return None

                if user_intent == "attribution":
                    if not user_input.isdigit():
                        if user_input in ["no", "NO", "No"]:
                            response = [
                                "no", "OK, great! It seems you are satisfied with current results. Please feel free to ask more questions : )"]
                            return response
                        else:
                            check_response = "Your input is not in correct format, please type only one number (e.g., 6) or 'No' as your input."
                            return check_response
                    else:
                        self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"] = {
                            "top_k": int(user_input)}
                        self.convxai_global_status_track[
                            f"intent_round_{self.convxai_intent_round}"]["binded"] = False
                        return None

                if user_intent == "counterfactual":
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

    def explain(self, explainInput, writingInput, predictOutputs, inputTexts, writingIndex, response_indicator=1, check_response=None, **kwargs):

        ####################################################################
        # ****** [Init-Conversation]: welcome sentences + Generate AI Comments ****** #
        ####################################################################
        if explainInput in ["ACL", "CHI", "ICLR"]:
            self.convxai_global_status_track["conference"] = explainInput
            self.ai_commenter = AICommenter(
                explainInput, writingInput, predictOutputs, inputTexts)
            response = self.ai_commenter.ai_comment_generation()
            response_init_xai = "Do you need <strong>some explanations</strong> of the above reviews?"
            return [response, response_init_xai], 2




        ####################################################################
        # ****** [User Intent Classification] ****** #
        ####################################################################
        explainInput = explainInput.strip().lower()

        if f"intent_round_{self.convxai_intent_round}" not in self.convxai_global_status_track.keys():
            ###### """Initiate the 'conversation_intent_round_k' dictionary """ ######
            self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"] = {
                "user_intent": None,
                "binded": False,
                "attributes": None
            }
        
        if self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["binded"] is True:
            user_intent_check = self.nlu(explainInput)
            if user_intent_check != "none":
                user_intent = user_intent_check
                if user_intent_check != self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]['user_intent']:
                    self.convxai_intent_round += 1
                    self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"] = {
                        "user_intent": None,
                        "binded": False,
                        "attributes": None
                    }
            else:
                user_intent = self.convxai_global_status_track[
                    f"intent_round_{self.convxai_intent_round}"]['user_intent']
        else:
            user_intent = self.nlu(explainInput)



        ####################################################################
        # ****** [Generating AI Explanations] ****** #
        ####################################################################
        predictLabel = [predictOutputs["outputs_predict_list"][int(k)] for k in writingIndex]

        ###### Generating Global AI Explanations ######
        if user_intent in ["ai-comment-global", "meta-data", "meta-model", "quality-score", "aspect-distribution", "sentence-length"]:
            
            if user_intent == "ai-comment-global":
                response = "<strong>-Basic Information and Statistics</strong> of the data and model for generating your reviews above:<br>"
                self.convxai_intent_round += 1
                return [response], 3

            explanations = self.explainer.generate_explanation(user_intent, "", "", self.convxai_global_status_track["conference"], self.global_explanations_data, **self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"])
            response = self.nlg(explanations)
            self.convxai_intent_round += 1
            return [response], response_indicator


        ###### Generating Local AI Explanations ######
        elif user_intent in ["ai-comment-instance", "confidence", "example", "attribution", "counterfactual"]:

            ###### Ensure the user select one sentence for explanations ######
            if len(writingInput) == 0:
                response = "You are asking for <strong>instance-wise explanations</strong>, and <strong>please select (double click) one sentence</strong> to be explained."
                return [response], response_indicator

            if user_intent == "ai-comment-instance":
                response = self.ai_commenter.explaining_ai_comment_instance(list(set(writingIndex)), self.convxai_global_status_track)
                self.convxai_intent_round += 1
                return response, 4


            ###### Collect the variables to generate explanations. ######
            check_response = self._check_multi_turn_input_variables(user_intent, explainInput)

            responses = []
            for k in range(len(writingInput)):

                ###### Stage3: Generating Explanations ######
                explanations = self.explainer.generate_explanation(user_intent, writingInput[k], predictLabel[k], self.convxai_global_status_track[
                                                                    "conference"], self.global_explanations_data, **self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"])
                response = self.nlg(explanations)
                responses.append(response)

            if check_response is not None:
                if len(check_response) == 2:
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

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


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

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
    """Based on AI predictions, AICommenter generate high-level AI comments that are more understandable and useful for users in practice."""
    def __init__(self, conference, writingInput, predictOutputs, inputTexts):
        self.conference = conference
        self.predictOutputs = predictOutputs
        self.writingInput = writingInput
        self.inputTexts = inputTexts
        self.review_summary = {}
        self.aspect_model = TfidfAspectModel(self.conference)
        self.revision_comment_template = {
            "shorter_length": "&nbsp;&nbsp;<p class='comments' id={id} class-id=shorter-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: The sentence is <strong>short than 20%</strong> of the published <strong>'{label}'</strong>-labeled sentences in {conference} conference. The average length is {ave_word} words.</p><br>",
            "longer_length": "&nbsp;&nbsp;<p class='comments' id={id} class-id=longer-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: The sentence is <strong>longer than 80%</strong> of the published <strong>'{label}'</strong>-labeled sentences in {conference} conference. The average length is {ave_word} words.</p><br>",
            "label_change": "&nbsp;&nbsp;<p class='comments' id={id} class-id=label-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: Based on the aspect <strong>labels' percentage and order</strong> of your abstract, it is suggested to write your <strong>{aspect_new}</strong> at this sentence, rather than describing <strong>{aspect_origin}</strong> here.</p><br>",
            "low_score": "&nbsp;&nbsp;<p class='comments' id={id} class-id=score-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: The style quality score of {sentence} is <strong>lower than 20%</strong> of the published <strong>'{label}'</strong>-labeled sentences in {conference} conference. Indicating the writing style might not match well with this conference.<br>"
        }
        self.global_explanations_data = global_explanations_data
            
    def _ai_comment_score_classifier(self, raw_score, score_benchmark):
        """The lower scores get higher labels."""
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

        # Aspects Labels Analysis ######: 
        # DTW algorithm: https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html#dtw-distance-measure-between-two-time-series;
        # https://towardsdatascience.com/an-illustrative-introduction-to-dynamic-time-warping-36aa98513b98
        distance, paths = dtw.warping_paths(
            aspect_list, aspect_distribution_benchmark)
        best_path = dtw.best_path(paths)
        change_label_dict = {}
        for path in best_path:
            if aspect_list[path[0]] != aspect_distribution_benchmark[path[1]]:
                change_label_dict[path[0]] = path[1]

        quality_score = 0
        ###### Length and Score Analysis ######
        for n in range(len(sentence_raw_score)):
            sentence_score = self._ai_comment_score_classifier(
                sentence_raw_score[n], sentence_score_benchmark[diversity_model_label_mapping[aspect_list[n]]])
            ### _ai_comment_score_classifier is 'lower score indicates longer sentence'.
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
        abstract_quality_score = quality_score / len(sentence_raw_score)

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

        abstract_structure_score = 5 - 0.5 * len(change_label_dict.keys())
        overall_score = (abstract_structure_score + abstract_quality_score) / 2
        print(f"===>>> The overall_score = {overall_score}, with abstract_structure_score = {abstract_structure_score} and abstract_quality_score = {abstract_quality_score}.")
        analysis_outputs = {
            "abstract_score": overall_score,
            "abstract_structure_score": abstract_structure_score,
            "abstract_quality_score": abstract_quality_score,
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
        comment_conference_intro = f"Nice! I'm comparing your abstract with <strong>{self.global_explanations_data[self.conference]['paper_count']} published {self.conference}</strong> abstracts."
        comment_overall_score = f"<br><br><p class='overall-score'> Your <strong>Overall Score</strong>=<strong>{analysis_outputs['abstract_score']:0.2f}</strong> (out of 5) by averaging: <br> Structure Score = {analysis_outputs['abstract_structure_score']:0.2f} and Style Score = {analysis_outputs['abstract_quality_score']:0.2f}. </p> "
        comment_improvements = "<br>" + analysis_outputs['instance_results'] if len(analysis_outputs['instance_results']) > 0 else "Your current writing looks good to me. Great job!"
        return comment_conference_intro + comment_overall_score + comment_improvements


    # def _explaining_ai_comment_template(self, idx, review, convxai_global_status_track):

    #     review_type = review.split("-")[0]

    #     if review_type == "long":

    #         response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: we are comparing the <strong>length</strong> of this sentence with all the sentences' in {convxai_global_status_track["conference"]} dataset.
    #         This sentence has <strong>{review.split("-")[1]} words</strong>, longer than 80% of the <strong>'{self.review_summary['prediction_label']}'</strong>-labeled sentences in the {convxai_global_status_track["conference"]} conference.
    #         <br><br>
    #         <strong>To improve</strong>, the sentence is suggested to have <strong>{self.review_summary['sentence_length_benchmark'][1]} to {self.review_summary['sentence_length_benchmark'][3]}</strong> words.
    #         <br><br>
    #         See more details of sentence length statistics, please click the question below:<br><br>
    #         """

    #     elif review_type == "short":
    #         response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: we are comparing the <strong>length</strong> of this sentence with all the sentences' in {convxai_global_status_track["conference"]} dataset.
    #         This sentence has <strong>{review.split("-")[1]} words</strong>, shorter than 20% of the <strong>'{self.review_summary['prediction_label']}'</strong>-labeled sentences in the {convxai_global_status_track["conference"]} conference.
    #         <br><br>
    #         <strong>To improve</strong>, the sentence is suggested to have <strong>{self.review_summary['sentence_length_benchmark'][1]} to {self.review_summary['sentence_length_benchmark'][3]}</strong> words.
    #         <br><br>
    #         See the more details of sentence length statistics, please click the question below:<br><br>
    #         """

    #     elif review_type == "quality":
    #         response = f"""
    #         <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: the <strong>Writing Style Model</strong> can generate a quality score, indicating <strong>how well the sentence can match with the {convxai_global_status_track["conference"]} conference</strong>, with <strong>lower the better</strong> style quality.
    #         <br><br>
    #         This sentence gets <strong>{float(review.split("-")[1]):.2f}</strong> points, which is larger than <strong>80%</strong> of the <strong>'{self.review_summary['prediction_label']}'</strong>-labeled sentences in the {convxai_global_status_track["conference"]} conference. Indicating the sentence may not match well with the {convxai_global_status_track["conference"]} conference. 
    #         <br><br>
    #         <strong>To improve</strong>, you can check similar sentences in {convxai_global_status_track["conference"]} to rewrite. Please click the question below:<br><br>
    #         """

    #     elif review_type == "aspect":
    #         response = f"""
    #         <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: we summarized five <strong>five structural patterns</strong> in {convxai_global_status_track["conference"]}, and found this abstract is closest to the pattern: <span class='text-danger font-weight-bold'>{self.review_summary["aspect_distribution_benchmark"]}</span>. 
    #         <br>
    #         <br>
    #         By comparing this abstract pattern with the closest benchmark using <a class='post-link' href='https://en.wikipedia.org/wiki/Dynamic_time_warping' target='_blank'><strong>Dynamic Time Warping</strong></a> algorithm,
    #         we suggest you to describe <strong>{review.split("-")[2]}</strong> aspect but <strong>not {review.split("-")[1]}</strong> in this sentence to improve the structure.
    #         <br><br>
    #         <strong>To improve</strong>, you can check the <strong> most important words </strong> for the label and further check <strong> how to revise input into another label</strong> . See XAI questions below:<br><br>
    #         """

    #     return [review_type, response]



    def _explaining_ai_comment_template(self, idx, review, convxai_global_status_track):

        review_type = review.split("-")[0]

        if review_type == "long":
            response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: XAI use patterns to <strong>shorten</strong> sentence length:
            <br><br>-<p style='color:#1B5AA2;font-weight:bold'>Pattern1: Similar Examples (rank: short)</p>. Refer to similar short examples for rewriting.  
            <br><br>-<p style='color:#1B5AA2;font-weight:bold'>Pattern2: Rewrite while keeping important_words</p>. Find important words, then keep them during rewriting to keep the correct aspects.
            """

            # response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: we are comparing the <strong>length</strong> of this sentence with all the sentences' in {convxai_global_status_track["conference"]} dataset.
            # This sentence has <strong>{review.split("-")[1]} words</strong>, longer than 80% of the <strong>'{self.review_summary['prediction_label']}'</strong>-labeled sentences in the {convxai_global_status_track["conference"]} conference.
            # <br><br>
            # <strong>To improve</strong>, the sentence is suggested to have <strong>{self.review_summary['sentence_length_benchmark'][1]} to {self.review_summary['sentence_length_benchmark'][3]}</strong> words.
            # <br><br>
            # See more details of sentence length statistics, please click the question below:<br><br>
            # """

        elif review_type == "short":
            response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: XAI use patterns to <strong>lengthen</strong> sentence length:
            <br><br>-<p style='color:#1B5AA2;font-weight:bold'>Pattern1: Similar Examples (rank: long)</p>. Refer to similar long examples for rewriting.  
            <br><br>-<p style='color:#1B5AA2;font-weight:bold'>Pattern2: Rewrite while keeping important_words</p>. Find important words, then keep them during rewriting to keep the correct aspects.
            """

            # response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: we are comparing the <strong>length</strong> of this sentence with all the sentences' in {convxai_global_status_track["conference"]} dataset.
            # This sentence has <strong>{review.split("-")[1]} words</strong>, shorter than 20% of the <strong>'{self.review_summary['prediction_label']}'</strong>-labeled sentences in the {convxai_global_status_track["conference"]} conference.
            # <br><br>
            # <strong>To improve</strong>, the sentence is suggested to have <strong>{self.review_summary['sentence_length_benchmark'][1]} to {self.review_summary['sentence_length_benchmark'][3]}</strong> words.
            # <br><br>
            # See the more details of sentence length statistics, please click the question below:<br><br>
            # """

        elif review_type == "quality":
            response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: XAI use patterns to <strong>improve sentence quality</strong>:
            <br><br>-<p style='color:#1B5AA2;font-weight:bold'>Pattern1: Counterfactual Explanation (use same label)</p>. Ask GPT-3 model to paraphrase the original sentence.  
            <br><br>-<p style='color:#1B5AA2;font-weight:bold'>Pattern2: Similar Examples (rank: quality_score).</p>. Refer to similar examples with high quality scores.
            """

            # response = f"""
            # <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: the <strong>Writing Style Model</strong> can generate a quality score, indicating <strong>how well the sentence can match with the {convxai_global_status_track["conference"]} conference</strong>, with <strong>lower the better</strong> style quality.
            # <br><br>
            # This sentence gets <strong>{float(review.split("-")[1]):.2f}</strong> points, which is larger than <strong>80%</strong> of the <strong>'{self.review_summary['prediction_label']}'</strong>-labeled sentences in the {convxai_global_status_track["conference"]} conference. Indicating the sentence may not match well with the {convxai_global_status_track["conference"]} conference. 
            # <br><br>
            # <strong>To improve</strong>, you can check similar sentences in {convxai_global_status_track["conference"]} to rewrite. Please click the question below:<br><br>
            # """

        elif review_type == "aspect":
            response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: XAI use patterns to <strong>rewrite into target-label</strong>:
            <br><br>-<p style='color:#1B5AA2;font-weight:bold'>Pattern1: Counterfactual Explanation (use target-label)</p>. Ask GPT-3 model to rewrite the sentence into the target aspect.  
            <br><br>-<p style='color:#1B5AA2;font-weight:bold'>Pattern2: Similar Examples (label: target-label, rank: quality_score).</p>. Refer to similar examples with the target labels and high quality scores.
            """

            # response = f"""
            # <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: we summarized five <strong>five structural patterns</strong> in {convxai_global_status_track["conference"]}, and found this abstract is closest to the pattern: <span class='text-danger font-weight-bold'>{self.review_summary["aspect_distribution_benchmark"]}</span>. 
            # <br>
            # <br>
            # By comparing this abstract pattern with the closest benchmark using <a class='post-link' href='https://en.wikipedia.org/wiki/Dynamic_time_warping' target='_blank'><strong>Dynamic Time Warping</strong></a> algorithm,
            # we suggest you to describe <strong>{review.split("-")[2]}</strong> aspect but <strong>not {review.split("-")[1]}</strong> in this sentence to improve the structure.
            # <br><br>
            # <strong>To improve</strong>, you can check the <strong> most important words </strong> for the label and further check <strong> how to revise input into another label</strong> . See XAI questions below:<br><br>
            # """

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

        # ****** Set up convxai_global_status_track to track the multi-turn variables ****** #
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
                            "top_k": 3, "aspect": None, "keyword": None, "rank": None}
                        # check_response = "Want to check <span class='text-danger font-weight-bold'>more examples of another label</span>? Please type <span class='text-danger font-weight-bold'>'word number + label'</span> (e.g., '6 + method'). <br><br> Or want to find <span class='text-danger font-weight-bold'>examples with keyword?</span>, please start with <span class='text-danger font-weight-bold'>'keyword:'</span> (e.g., 'keyword: robotics')</span>. <br><br>Otherwise, please reply '<span class='text-danger font-weight-bold'>No</span>'."
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
                        
                        # <span class='text-danger font-weight-bold'>'word number + label'</span> (e.g., '6 + method'). <br><br> Or want to find <span class='text-danger font-weight-bold'>examples with keyword?</span>, please start with <span class='text-danger font-weight-bold'>'keyword:'</span> (e.g., 'keyword: robotics')</span>. <br><br>Otherwise, please reply '<span class='text-danger font-weight-bold'>No</span>'.


                if user_intent == "attribution":
                    if self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"] is None:
                        self.convxai_global_status_track[
                            f"intent_round_{self.convxai_intent_round}"]["binded"] = True
                        self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"] = {
                            "top_k": 3, "aspect": None}
                        check_response = "Want to highlight <span class='text-danger font-weight-bold'>more important words for another label</span>? Please type <span class='text-danger font-weight-bold'>'word number + label'</span> (e.g., '6 + method'), <br><br> Or just say '<span class='text-danger font-weight-bold'>No</span>' if you don't need more examples : )."
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
                    ### Parse the user input variables in text.
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
                        ### Add key values into variable dictionary.
                        user_input_dict = {}
                        all_tokens = []
                        for k in range(len(key_list)):
                            all_tokens.extend(key_list[k].replace(",", " ").split())
                            
                        for t in range(len(all_tokens)):
                            if all_tokens[t] in ['rank', 'count', 'label', 'keyword']:
                                value = all_tokens[t+1]
                                user_input_dict[all_tokens[t]] = value
                        try:
                            if 'count' in user_input_dict.keys():
                                self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"]["top_k"] = int(user_input_dict['count'])
                            if 'label' in user_input_dict.keys():
                                self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"]["aspect"] = user_input_dict['label']
                            if 'keyword' in user_input_dict.keys():
                                self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"]["keyword"] = user_input_dict['keyword']
                            if 'rank' in user_input_dict.keys():
                                self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"]["rank"] = user_input_dict['rank']
                            return "continue"
                        except:
                            check_response = """Your input is not in correct format, 
                                <br><br> Please specify ONE or MORE of them by
                                <span class='font-weight-bold'>'label:your_label, keyword:your_keyword, rank:your_rank_method, count:your_top_k'</span>
                                <br>(e.g., <span class='text-danger font-weight-bold'>'count:8'</span>, or <span class='text-danger font-weight-bold'>'label:background, keyword:time, rank:quality_score, count:6'</span>)."""
                            return check_response

                if user_intent == "attribution":
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
                            self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"]["top_k"] = int(user_input)
                            self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["attributes"]["aspect"] = aspect
                            return "continue"

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
                    ### NEW Added
                    self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"]["binded"] = False
                    ###
            else:
                user_intent = self.convxai_global_status_track[
                    f"intent_round_{self.convxai_intent_round}"]['user_intent']
        else:
            user_intent = self.nlu(explainInput)

        print(f"===> Detected User Intent = {user_intent}")

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

            ###### Explaining the AI comments. ######
            if user_intent == "ai-comment-instance":
                response = self.ai_commenter.explaining_ai_comment_instance(list(set(writingIndex)), self.convxai_global_status_track)
                self.convxai_intent_round += 1
                return response, 4

            ###### Collect the variables to generate explanations. ######
            check_response = self._check_multi_turn_input_variables(user_intent, explainInput)
            responses = []
            for k in range(len(writingInput)):
                explanations = self.explainer.generate_explanation(user_intent, writingInput[k], predictLabel[k], self.convxai_global_status_track[
                                                                    "conference"], self.global_explanations_data, **self.convxai_global_status_track[f"intent_round_{self.convxai_intent_round}"])
                response = self.nlg(explanations)
                responses.append(response)

            if check_response is not None:
                if check_response == "continue":
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

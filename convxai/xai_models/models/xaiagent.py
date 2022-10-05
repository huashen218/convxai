from ast import Pass
from curses.ascii import isdigit
import os
import json
from tkinter import TRUE
import torch
import numpy as np
import torch.nn.functional as F
from parlai.core.agents import register_agent, Agent
from .modules import *
from convxai.writing_models.utils import *

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


import logging
from collections import defaultdict
from dtaidistance import dtw



logger = logging.getLogger("my-logger")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT)
logger.setLevel(logging.INFO)





from pathlib import Path
import joblib
import json
import os
from typing import List
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans



label_mapping_inverse = {
    0: "background",
    1: "purpose",
    2: "method", 
    3: "finding",
    4: "other",
}

label_mapping = {
    "background": 0,
    "purpose": 1,
    "method": 2, 
    "finding": 3,
    "other": 4,
}



@dataclass
class AspectPattern:
    _id: str
    aspect_sequence: List[int]


def tokenize(x):
    return x.split(" ")
    

class TfidfAspectModel:
    # def __init__(self, model_path: Path):
    def __init__(self, conference):
        model_path = Path(
        "/data/hua/workspace/projects/convxai/src/convxai/xai_models/checkpoints/xai_writing_aspect_prediction",
        f"{conference}",
        )

        self.vectorizer = joblib.load(Path(model_path, "vectorizer.joblib"))
        self.model = joblib.load(Path(model_path, "model.joblib"))
        with open(Path(model_path, "centers.json"), 'r', encoding='utf-8') as infile:
            self.centers = json.load(infile)

    def get_feature(self, aspect_sequence: List[int]):
        return " ".join([
            f"{aspect}"
            for i, aspect in enumerate(aspect_sequence)
        ])

    def predict(self, aspect_sequence: List[int]) -> AspectPattern:
        feature = self.get_feature(aspect_sequence)
        vectors = self.vectorizer.transform([feature])
        labels = self.model.predict(vectors)
        result = AspectPattern(*self.centers[labels[0]])
        return result



@register_agent("xai")
class XaiAgent(Agent):


    @classmethod
    def add_cmdline_args(cls, parser, partial_opt):
        parser.add_argument('--name', type=str, default='Alice', help="The agent's name.")
        return parser


    def __init__(self, opt, shared=None):
        # similar to the teacher, we have the Opt and the shared memory objects!
        super().__init__(opt, shared)
        self.id = 'XaiAgent'
        self.name = opt['name']
        self.xai_agent = XaiExplainer()



    def observe(self, observation):
        self.input = observation if observation else  "Invalid input. Please ask your question again."

        outputs_predict = self.input["predict_results"]['diversity_model_outputs']['outputs_predict']
        outputs_predict_list = [int(v) for v in outputs_predict.split(',')]

        outputs_probability = self.input["predict_results"]['diversity_model_outputs']['outputs_probability']
        outputs_probability_list = [float(v) for v in outputs_probability.split(',')]

        outputs_perplexity = self.input["predict_results"]['quality_model_outputs']['outputs_perplexity']
        outputs_perplexity_list = [float(v) for v in outputs_perplexity.split(',')]

        self.predictOutputs = {
            "outputs_predict_list": outputs_predict_list,
            "outputs_probability_list": outputs_probability_list,
            "outputs_perplexity_list": outputs_perplexity_list
        }


    def act(self):
        """Parse the self.input information to get the 'explainInput', 'writingInput' and 'writingModel'.
        self.input = {
            "message_type": 'conv',
            "text": '{"writingInput": 'List[str]', "explainInput": 'str', "writingIndex": 'List[str]'}',
            "writing_model": 'model-writing-1',
            "message_id": 'ObjectId('622ab03276bdb0ec3f36cb18')',
        }
        """

        writingInput = json.loads(self.input["text"])["writingInput"]
        writingModel = "diversity_model" if self.input["writing_model"] == 'model-writing-1' else "quality_model"
        explainInput = json.loads(self.input["text"])["explainInput"]
        writingIndex = json.loads(self.input["text"])["writingIndex"]
        ButtonID     = json.loads(self.input["text"])["ButtonID"]

        inputTexts = self.input["predict_results"]["diversity_model_outputs"]["inputs"]

        response_text, response_indicator = self.xai_agent.explain(explainInput, writingInput, writingModel, self.predictOutputs, inputTexts, writingIndex, ButtonID)
        response = {
            "text": response_text,
            "writingIndex": response_indicator,
            "responseIndicator": [response_indicator],
        }

        return response








class XaiExplainer(object):
    """XaiExplainer
    """

    def __init__(self, woz_setting=False) -> None:
        self.woz_setting = woz_setting
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        with open(os.path.join(root_dir, "xai_models/configs/xai_model_configs.json" ), 'r') as fp:
            self.xai_model_configs = json.load(fp)
        # self.nlu = XAI_NLU_Module(self.xai_model_configs, nlu_algorithm = "scibert_classifier")
        self.nlu = XAI_NLU_Module(self.xai_model_configs, nlu_algorithm = "rule_based")
        self.explainer = Model_Explainer(self.xai_model_configs)
        self.nlg = XAI_NLG_Module()

        self.multiturn_intent_rounds = defaultdict()
        self.intent_round = 0
        self.multi_turn_required_intent_list = ["example", "attribution", "counterfactual"] 
        self.review_summary = {}

        self.writing_feedback_template = {
            "shorter_length": "&nbsp;&nbsp;<p class='comments' id={id} class-id=shorter-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: The sentence is <strong>too short</strong>, the average length of the sentences predicted as <strong>'{label}'</strong> labels in {conference} conference is {ave_word} words. Please rewrite it into a longer one.</p><br>",
            "longer_length": "&nbsp;&nbsp;<p class='comments' id={id} class-id=longer-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: The sentence is <strong>too long</strong>, the average length of the sentences predicted as <strong>'{label}'</strong> labels in {conference} conference is {ave_word} words. Please rewrite it into a shorter one.</p><br>",
            # "label_change": "&nbsp;&nbsp;<p class='comments' id={id} class-id=label-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: It is suggested to write your <strong>{aspect_new}</strong> in this sentence, rather than describing <strong>{aspect_origin}</strong> in the current abstract.</p><br>",
            "label_change": "&nbsp;&nbsp;<p class='comments' id={id} class-id=label-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: Based on the sentence <strong>labels' percentage and order</strong> in your abstract, it is suggested to write your <strong>{aspect_new}</strong> at this sentence, rather than describing <strong>{aspect_origin}</strong> here.</p><br>",
            "low_score": "&nbsp;&nbsp;<p class='comments' id={id} class-id=score-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: The writing style quality score of {sentence} is a bit <strong>lower</strong> than <strong>'{label}'</strong>-labeled sentences in the {conference} conference. This indicate the writing style might not match well with this conference.<br>"
        }

        ###### data_stats = [-2*std, -std, mean, +std, +2*std]
        self.global_explanations_data = {
            "ACL": {
                "paper_count": 3221,
                "sentence_count": 20744,
                "sentence_length": 
                    {
                        "all":        [2, 14, 26, 38, 50],
                        "background": [4, 14, 23, 33, 42],
                        "purpose":    [6, 16, 27, 37, 47],
                        "method":     [0, 14, 27, 41, 55],
                        "finding":    [1, 14, 26, 39, 52],
                        "other":      [-11,-3,5,  14, 22]
                    },
                "sentence_score_range": 
                    {
                        "all": [22, 32, 39, 46, 71],
                        "background": [21, 31, 36, 44, 68],
                        "purpose": [19, 26, 30, 35, 51],
                        "method": [27, 38, 44, 53, 78],
                        "finding": [22, 33, 40, 48, 76],
                        "other": [36, 63, 102, 213, 692]
                    },
                # "abstract_score_range": [41, 51, 57, 63, 80],
                "abstract_score_range": [32, 41, 46, 51, 67],
                "aspect_distribution": [],
                "Aspect_Patterns_dict": {
                    "00122233": "'background' (25%)   -&gt; 'purpose' (12.5%) -&gt; 'method'  (37.5%) -&gt; 'finding' (25%)",
                    "001233": "'background' (33.3%) -&gt; 'purpose' (16.7%) -&gt; 'method'  (16.7%) -&gt; 'finding' (33.3%)",
                    "0002233": "'background' (42.9%) -&gt; 'method'  (28.6%) -&gt; 'finding' (28.5%)",
                    "000133": "'background' (50%)   -&gt; 'purpose' (16.7%) -&gt; 'finding' (33.3%)",
                    "00323333": "'background' (25%)   -&gt; 'finding' (12.5%) -&gt; 'method'  (12.5%) -&gt; 'finding' (50%)",
                },
            },
            "CHI": {
                "paper_count": 3235,
                "sentence_count": 21643,
                "sentence_length": 
                    {
                        "all":        [4, 14, 25, 36, 46],
                        "background": [5, 14, 22, 31, 40],
                        "purpose":    [6, 17, 27, 38, 49],
                        "method":     [4, 15, 27, 39, 51],
                        "finding":    [4, 15, 26, 37, 48],
                        "other":      [-3,0,  4,  7,  11]
                    },
                "sentence_score_range": 
                    {
                        "all": [32, 45, 53, 63, 97],
                        "background": [28, 40, 48, 57, 88],
                        "purpose": [28, 39, 45, 52, 75],
                        "method": [33, 47, 56, 67, 103],
                        "finding": [37, 52, 62, 73, 111],
                        "other": [21, 117, 177, 523, 5979]
                    },
                # "abstract_score_range": [62, 78,  86, 96, 125],
                "abstract_score_range": [45, 57,  63, 71, 92],
                "aspect_distribution": [],
                "Aspect_Patterns_dict": {
                    "0001333": "'background' (42.9%) -&gt; 'purpose' (14.3%)  -&gt; 'finding' (42.9%)",
                    "001222333": "'background' (22.2%) -&gt; 'purpose' (11.2%) -&gt; 'method' (33.3%) -&gt; 'finding' (33.3%)",
                    "001233": "'background' (33.3%) -&gt; 'purpose' (16.7%) -&gt; 'method' (16.7%)  -&gt; 'finding' (33.3%)",
                    "002333": "'background' (33.3%) -&gt; 'method' (16.7%)  -&gt;  'finding' (50%)",
                    "000300100323333": "'background' (20%)   -&gt; 'finding' (6.7%)  -&gt;  'background' (13.3%) -&gt; 'purpose' (6.7%) -&gt; 'background' (13.3%) -&gt; 'finding' (6.7%) -&gt; 'method' (6.7%) -&gt; 'finding' (26.7%)"
                },
            },
            "ICLR": {
                "paper_count": 3479,
                "sentence_count": 25873,
                "sentence_length": 
                    {
                        "all":        [0, 13, 27, 41, 54],
                        "background": [2, 13, 24, 36, 47],
                        "purpose":    [4, 16, 28, 39, 51],
                        "method":     [-2,13, 29, 45, 61],
                        "finding":    [0, 14, 28, 42, 55],
                        "other":      [-1,2,  5,  8,  11]
                    },
                "sentence_score_range": 
                    {
                        "all": [35, 52, 62, 74, 116],
                        "background": [32, 48, 57, 68, 106],
                        "purpose": [28, 40, 47, 56, 83],
                        "method": [40, 58, 68, 82, 125],
                        "finding": [37, 56, 66, 80, 126],
                        "other": [31, 50, 59, 79, 371]
                    },
                # "abstract_score_range": [68, 85, 96, 107, 141],
                "abstract_score_range": [52, 67,  76, 84, 111],
                "aspect_distribution": [],
                "Aspect_Patterns_dict": {
                    "001233": "'background' (33.3%) -&gt; 'purpose' (16.7%) -&gt; 'method' (16.7%) -&gt; 'finding' (33.3%)",
                    "23333": "'Method' (20%) -&gt; 'finding' (80%)",
                    "0001333": "'background' (42.9%) -&gt; 'purpose' (14.2) -&gt; 'finding' (42.9%)",
                    "00000232333": "'background' (45.5%) -&gt; 'method' (9.1%) -&gt; 'finding' (9.1%) -&gt; 'method' (9.1%) -&gt; 'finding' (27.3%)",
                    "001222333": "'Background' (22.2%) -&gt; 'purpose' (11.1%) -&gt; 'method' (33.3%) -&gt; 'finding' (33.4%)",
                }
            },
            "Aspect_Patterns-ACL":[
                "'background' (25%)   -&gt; 'purpose' (12.5%) -&gt; 'method'  (37.5%) -&gt; 'finding' (25%)",
                "'background' (33.3%) -&gt; 'purpose' (16.7%) -&gt; 'method'  (16.7%) -&gt; 'finding' (33.3%)",
                "'background' (42.9%) -&gt; 'method'  (28.6%) -&gt; 'finding' (28.5%)",
                "'background' (50%)   -&gt; 'purpose' (16.7%) -&gt; 'finding' (33.3%)",
                "'background' (25%)   -&gt; 'finding' (12.5%) -&gt; 'method'  (12.5%) -&gt; 'finding' (50%)",
            ],
            "Aspect_Patterns-CHI":[
                "'background' (42.9%) -&gt; 'purpose' (14.3%)  -&gt; 'finding' (42.9%)",
                "'background' (22.2%) -&gt; 'purpose' (11.2%) -&gt; 'method' (33.3%) -&gt; 'finding' (33.3%)",
                "'background' (33.3%) -&gt; 'purpose' (16.7%) -&gt; 'method' (16.7%)  -&gt; 'finding' (33.3%)",
                "'background' (33.3%) -&gt; 'method' (16.7%)  -&gt;  'finding' (50%)",
                "'background' (20%)   -&gt; 'finding' (6.7%)  -&gt;  'background' (13.3%) -&gt; 'purpose' (6.7%) -&gt; 'background' (13.3%) -&gt; 'finding' (6.7%) -&gt; 'method' (6.7%) -&gt; 'finding' (26.7%)"
            ],
            "Aspect_Patterns-ICLR":[
                "'background' (33.3%) -&gt; 'purpose' (16.7%) -&gt; 'method' (16.7%) -&gt; 'finding' (33.3%)",
                "'Method' (20%) -&gt; 'finding' (80%)",
                "'background' (42.9%) -&gt; 'purpose' (14.2) -&gt; 'finding' (42.9%)",
                "'background' (45.5%) -&gt; 'method' (9.1%) -&gt; 'finding' (9.1%) -&gt; 'method' (9.1%) -&gt; 'finding' (27.3%)",
                "'Background' (22.2%) -&gt; 'purpose' (11.1%) -&gt; 'method' (33.3%) -&gt; 'finding' (33.4%)",
            ]
        }




    def _score_classifier(self, raw_score, score_benchmark):
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


    def writing_analyzer(self, predictOutputs, conference, inputTexts):

        structure_score, quality_score = 5, 0

        ###### Benchmark Scores ######
        abstract_score_benchmark = self.global_explanations_data[conference]['abstract_score_range']
        sentence_score_benchmark  = self.global_explanations_data[conference]['sentence_score_range']  ### quality_score_feedback
        # sentence_length_benchmark = self.global_explanations_data[conference]['sentence_length']["all"]
        sentence_length_benchmark = self.global_explanations_data[conference]['sentence_length']

        ###### Quality Scores ######
        sentence_raw_score = predictOutputs["outputs_perplexity_list"]
        abstract_score = self._score_classifier(np.mean(sentence_raw_score), abstract_score_benchmark)


        ###### Aspects Labels & Patterns ######
        aspect_list = predictOutputs["outputs_predict_list"]    ### aspect_feedback
        aspect_distribution_benchmark = self.aspect_model.predict(aspect_list)
        aspect_distribution_benchmark = aspect_distribution_benchmark.aspect_sequence

        length_revision = ""
        structure_revision = ""
        style_revision = ""


        ###### Aspects Labels Analysis ######: DTW algorithm: https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html#dtw-distance-measure-between-two-time-series;   https://towardsdatascience.com/an-illustrative-introduction-to-dynamic-time-warping-36aa98513b98
        distance, paths = dtw.warping_paths(aspect_list, aspect_distribution_benchmark)
        best_path = dtw.best_path(paths)
        change_label_dict = {}
        for path in best_path:
            if aspect_list[path[0]] != aspect_distribution_benchmark[path[1]]:
                # change_label_dict.setdefault(path[0], []).append(path[1])
                change_label_dict[path[0]] = path[1]
                

        ###### Length and Score Analysis ######
        print("===>>>>>>>!!!sentence_raw_score", sentence_raw_score)
        for n in range(len(sentence_raw_score)):
        # for n in range(len(sentence_raw_score)+1):
            sentence_score = self._score_classifier(sentence_raw_score[n], sentence_score_benchmark[label_mapping_inverse[aspect_list[n]]])
            # sentence_length =  self._score_classifier(len(counting_tokens(inputTexts[n])), sentence_length_benchmark)
            sentence_length =  self._score_classifier(len(counting_tokens(inputTexts[n])), sentence_length_benchmark[label_mapping_inverse[aspect_list[n]]])


            if n in change_label_dict.keys():
                # for k in change_label_dict[n]:
                k = change_label_dict[n]
                structure_revision += self.writing_feedback_template["label_change"].format(id = f"{n}", sentence = f"S{n+1}", aspect_origin=label_mapping_inverse[aspect_list[n]], aspect_new = label_mapping_inverse[aspect_distribution_benchmark[k]])
                self.review_summary.setdefault(n, []).append(f"aspect-{label_mapping_inverse[aspect_list[n]]}-{label_mapping_inverse[aspect_distribution_benchmark[k]]}")

            if sentence_score < 2:
                style_revision += self.writing_feedback_template["low_score"].format(id = f"{n}", sentence = f"S{n+1}", label = label_mapping_inverse[aspect_list[n]], conference = conference)
                self.review_summary.setdefault(n, []).append(f"quality-{sentence_raw_score[n]}")
            
            if sentence_length > 4:  
                length_revision += self.writing_feedback_template["shorter_length"].format(id = f"{n}", sentence = f"S{n+1}", conference = conference, label = label_mapping_inverse[aspect_list[n]], ave_word=self.global_explanations_data[conference]["sentence_length"][label_mapping_inverse[aspect_list[n]]][2])
                self.review_summary.setdefault(n, []).append(f"short-{len(counting_tokens(inputTexts[n]))}")

            if sentence_length < 2: 
                length_revision += self.writing_feedback_template["longer_length"].format(id = f"{n}", sentence = f"S{n+1}", conference = conference, label = label_mapping_inverse[aspect_list[n]], ave_word=self.global_explanations_data[conference]["sentence_length"][label_mapping_inverse[aspect_list[n]]][2])
                self.review_summary.setdefault(n, []).append(f"long-{len(counting_tokens(inputTexts[n]))}")

            quality_score += sentence_score


            # self.review_summary["sentence_score_benchmark"][n] = self.global_explanations_data[conference]['sentence_score_range'][label_mapping_inverse[aspect_list[n]]]
            # self.review_summary["sentence_length_benchmark"][n] = self.global_explanations_data[conference]['sentence_length'][label_mapping_inverse[aspect_list[n]]]
            # self.review_summary["prediction_label"][n] = label_mapping_inverse[aspect_list[n]]

        print("===!!!!!!aspect_distribution_benchmarkaspect_distribution_benchmark", aspect_distribution_benchmark)

        self.review_summary["abstract_score_benchmark"] = self.global_explanations_data[conference]['abstract_score_range']
        self.review_summary["sentence_score_benchmark"] = self.global_explanations_data[conference]['sentence_score_range'][label_mapping_inverse[aspect_list[n]]]
        self.review_summary["sentence_length_benchmark"] = self.global_explanations_data[conference]['sentence_length'][label_mapping_inverse[aspect_list[n]]]
        self.review_summary["prediction_label"] = label_mapping_inverse[aspect_list[n]]
        aspect_keys = "".join(list(map(str, aspect_distribution_benchmark)))
        self.review_summary["aspect_distribution_benchmark"] = self.global_explanations_data[conference]["Aspect_Patterns_dict"][aspect_keys]


        if len(length_revision) == 0:
            length_revision = "&nbsp;&nbsp;<p><strong>-</strong>Your sentence lengths look good to me. Great job!</p>"

        if len(structure_revision) == 0:
            structure_revision = "&nbsp;&nbsp;<p><strong>-</strong>Your abstract structures look good to me. Great job!</p>"
        
        if len(style_revision) == 0:
            style_revision = "&nbsp;&nbsp;<p><strong>-</strong>Your writing styles look good to me. Great job!</p>"

        feedback_improvement = f"<br> <p style='color:#1B5AA2;font-weight:bold'> Structure Suggestions:</p> {structure_revision}" + \
                               f"<br> <p style='color:#1B5AA2;font-weight:bold'> Style Suggestions:</p> {style_revision} {length_revision}"

        structure_score = 5 - 0.5 * len(change_label_dict.keys())
        # overall_score = (structure_score + quality_score) / (len(sentence_raw_score) + 1)
        overall_score = (structure_score + quality_score / len(sentence_raw_score)) / 2

        analysis_outputs = {
            "abstract_score": overall_score,
            "instance_results": feedback_improvement
        }
        
        return  analysis_outputs



    def writing_feedback(self, explainInput, writingInput, predictOutputs, inputTexts):

        ###### Info Collection Results ######
        # response = f"Nice! I'm comparing your submission with <strong>{self.global_explanations_data[explainInput]['paper_count']}</strong> {explainInput} paper abstracts."
        response = f"Nice! I'm comparing your submission with {self.global_explanations_data[explainInput]['paper_count']} {explainInput} paper abstracts."


        ###### Overall Quality Score ######
        analysis_outputs = self.writing_analyzer(predictOutputs, explainInput, inputTexts)
        response_overall_score = f"<br><br><p class='overall-score'> Your <strong>Overall Score</strong> of Structure and Style = <strong>{analysis_outputs['abstract_score']:0.0f}</strong> (out of 5).</p> "
        
        ###### Instance-wise Score ######
        feedback_improvement = analysis_outputs['instance_results']
        if len(feedback_improvement) > 0:
            response_improvement = "<br>" + feedback_improvement
        else:
            response_improvement = "Your current writing looks good to me. Great job!"

        # return response + response_improvement
        return response + response_overall_score + response_improvement




    def _instance_explanation_template(self, idx, review):

        review_type = review.split("-")[0]

        if review_type == "long":

            response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: we are comparing the <strong>length</strong> of this sentence with all the sentences' in {self.multiturn_intent_rounds["conference"]} dataset.
            Your sentence has <strong>{review.split("-")[1]} words</strong>, which is longer than 80% of the <strong>'{self.review_summary['prediction_label']}'</strong>-labeled sentences in the {self.multiturn_intent_rounds["conference"]} conference.
            <br><br>
            <strong>To improve</strong>, the sentence is suggested to have <strong>{self.review_summary['sentence_length_benchmark'][1]} to {self.review_summary['sentence_length_benchmark'][3]}</strong> words.
            <br><br>
            To see the more details of sentence length statistics, please click the question below:<br><br>
            """
            # <strong>To improve</strong>, the sentence is suggested to have <strong>{self.review_summary['sentence_length_benchmark'][idx][1]} to {self.review_summary['sentence_length_benchmark'][idx][3]}</strong> words.

        elif review_type == "short":
            response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: we are comparing the <strong>length</strong> of this sentence with all the sentences' in {self.multiturn_intent_rounds["conference"]} dataset.
            This sentence has <strong>{review.split("-")[1]} words</strong>, which is shorter than 80% of the <strong>'{self.review_summary['prediction_label']}'</strong>-labeled sentences in the {self.multiturn_intent_rounds["conference"]} conference.
            <br><br>
            <strong>To improve</strong>, the sentence is suggested to have <strong>{self.review_summary['sentence_length_benchmark'][1]} to {self.review_summary['sentence_length_benchmark'][3]}</strong> words.
            <br><br>
            To see the more details of sentence length statistics, please click the question below:<br><br>
            """
            # <strong>To improve</strong>, the sentence is suggested to have <strong>{self.review_summary['sentence_length_benchmark'][idx][1]} to {self.review_summary['sentence_length_benchmark'][idx][3]}</strong> words.


        elif review_type == "quality":
            response = f"""
            <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: the <strong>Writing Style Model</strong> can generate a quality score, indicating <strong>how well the sentence can match with the {self.multiturn_intent_rounds["conference"]} conference</strong>, with <strong>lower the better</strong> style quality.
            <br><br>
            This sentence gets <strong>{float(review.split("-")[1]):.2f}</strong> points, which is larger than <strong>80%</strong> of the <strong>'{self.review_summary['prediction_label']}'</strong>-labeled sentences in the {self.multiturn_intent_rounds["conference"]} conference. This indicates the sentence' writing style might not match well with published {self.multiturn_intent_rounds["conference"]} sentences. 
            <br><br>
            <strong>To improve</strong>, you can check {self.multiturn_intent_rounds["conference"]} similar sentences for reference. Please click the question below:<br><br>
            """


        elif review_type == "aspect":
            response = f"""
            <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: we summarized all the collected {self.multiturn_intent_rounds["conference"]} abstracts into <strong>five structural patterns</strong>, where we found your submission is closest to the pattern of <span class='text-danger font-weight-bold'>{self.review_summary["aspect_distribution_benchmark"]}</span>. By using <a class='post-link' href='https://en.wikipedia.org/wiki/Dynamic_time_warping' target='_blank'><strong>Dynamic Time Warping</strong></a> algorithm to analyze <strong> how to revise your submission to fit this style pattern</strong>,
            the result suggested to describe <strong>{review.split("-")[2]}</strong> aspect but not <strong>{review.split("-")[1]}</strong> in this sentence.
            <br><br>
            <strong>To improve</strong>, you can check the <strong> most important words </strong> resulting in the prediction and further check <strong> how to revise input into another label</strong> . See XAI questions below:<br><br>
            """

        return [review_type, response]


    def instance_explanation(self, writingIndex):
        print("!!==========instance_explanation!!!writingIndex", writingIndex)
        print("===!!~~self.review_summary", self.review_summary)
        response = []
        for idx in writingIndex:
            if int(idx) in self.review_summary.keys():
                for item in self.review_summary[int(idx)]:

                    response.extend(self._instance_explanation_template(int(idx), item))
        if len(response) == 0:
            response = ["none", "Your writing looks good to us! <br><br><strong>To improve</strong>, you can ask for explanation questions below:<br><br>"]  # generate understanding here.
        return response


    def global_explanation(self):
        global_xai = "<strong>-Basic Information and Statistics</strong> of the data and model for generating your reviews above:<br>"
        response = [global_xai]
        return response


    def local_explanation(self):
        local_xai = "-Explanations for <strong>Each Sentence Prediction</strong>: please <strong>click the specific sentence</strong>, then click the below buttons for your questions."
        response = [local_xai]
        return response



    def check_user_input_variables(self, user_intent, user_input):

        self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["user_intent"] = user_intent

        if user_intent in self.multi_turn_required_intent_list:

            if self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["binded"] is False:
                if user_intent == "example":
                    if self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["attributes"] is None:
                        self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["binded"] = True
                        self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["attributes"] = {"top_k": 3, "aspect": None}
                        check_response = "Would you like to <span class='text-danger font-weight-bold'>see more or less examples</span>, and meanwhile conditioned on an aspect? If you need, please type the <span class='text-danger font-weight-bold'>word number + aspect (e.g., 6 + method)</span>, otherwise, please reply 'No'."
                        return check_response

                if user_intent == "attribution":
                    if self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["attributes"] is None:
                        self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["binded"] = True
                        self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["attributes"] = {"top_k": 3}
                        check_response = "Would you like to <span class='text-danger font-weight-bold'>highlight more or less important words</span>? If you need, please type the word number (e.g., 6), otherwise, please reply 'No'."
                        return check_response

                if user_intent == "counterfactual":
                    if self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["attributes"] is None:
                        self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["binded"] = True
                        self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["attributes"] = {"contrast_label": None}
                        check_response = "Would you like to <span class='text-danger font-weight-bold'>set another contrastive label</span> to change to? Please type the label from <span class='text-danger font-weight-bold'>'background', 'method', 'purpose', 'finding', 'others'</span>, or reply 'No' if you won't need."
                        return check_response


            elif self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["binded"] is True:
                
                user_input = str(user_input).lower().strip()
                user_input = user_input.replace(" ", "")

                if user_intent == "example":

                    if "+" in user_input:
                        user_input, aspect = user_input.split("+")
                        if not user_input.isdigit() or aspect not in ["background", "purpose", "method", "finding", "other"]:
                            if user_input in ["no", "NO", "No"]:
                                response = ["no", "OK, great! It seems you are satisfied with current results. Please feel free to ask more questions : )"]
                                return response
                            else:
                                check_response = "Your input is not in correct format, please type the word number + aspect (e.g., 6 + method), otherwise, please reply 'No'."
                                return check_response
                        else:
                            self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["attributes"] = {"top_k": int(user_input), "aspect": aspect}
                            self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["binded"] = False
                            return None


                if user_intent == "attribution":
                    if not user_input.isdigit():
                        if user_input in ["no", "NO", "No"]:
                            response = ["no", "OK, great! It seems you are satisfied with current results. Please feel free to ask more questions : )"]
                            return response
                        else:
                            check_response = "Your input is not in correct format, please type only one number (e.g., 6) or 'No' as your input."
                            return check_response
                    else:
                        self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["attributes"] = {"top_k": int(user_input)}
                        self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["binded"] = False
                        return None


                if user_intent == "counterfactual":
                    # if user_input not in ["background", "purpose", "method", "finding", "other"] or user_input in ["no", "NO", "No"]:
                    if user_input not in ["background", "purpose", "method", "finding", "other"]:
                        if user_input in ["no", "NO", "No"]:
                            response = ["no", "OK, great! It seems you are satisfied with current results. Please feel free to ask more questions : )"]
                            return response
                        else:
                            check_response = "Your input is not a correct label, please type one from <strong>['background', 'purpose', 'method', 'finding', 'other']</strong> below."
                            return check_response
                    else:
                        self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["attributes"] = {"contrast_label": user_input}
                        self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["binded"] = False
                        return None


        elif user_intent not in self.multi_turn_required_intent_list:
            return None




    def explain(self, explainInput, writingInput, writingModel, predictOutputs, inputTexts, writingIndex, ButtonID, response_indicator=1, check_response=None, **kwargs):   

        ###### Step1: [Welcome Tutorial] + [Summarized Writing Feedback] ######
        if explainInput in ["ACL", "CHI", "ICLR"]:
            self.aspect_model = TfidfAspectModel(explainInput)
            response = self.writing_feedback(explainInput, writingInput, predictOutputs, inputTexts)
            response_ask_for_xai = "Do you need <strong>some explanations</strong> of the above reviews?"
            self.multiturn_intent_rounds["conference"] = explainInput
            return [response, response_ask_for_xai], 2


        ###### Step2: [Explanation on Feedback] ######
        if "show me model and data" in explainInput:   ### "[GlobalWhy]"
            response = self.global_explanation()
            return response, 3



        explainInput = explainInput.strip().lower()
        # if len(list(set([explainInput]).intersection(set(["explain this sentence", "explain the sentence", "explain sentence review", "explain the review", "explain this review"])))) > 0:
        if "explain this sentence" in explainInput or \
            "explain the sentence" in explainInput or \
            "explain sentence review" in explainInput or \
            "explain this review" in explainInput or  \
            "explain the review" in explainInput:   ### "[InstanceWhy]"

            ###### Ensure the user select one sentence for explanations ######
            if len(writingIndex) == 0:
                response = "You are asking for <strong>instance-wise explanations</strong>, and <strong>please select (double click) one sentence</strong> to be explained."
                return [response], response_indicator
            else:
                response = self.instance_explanation(list(set(writingIndex)))
                return response, 4


        ###### Step3: [Explanation on AI Prediction] ######
        else:
            predictLabel = [predictOutputs["outputs_predict_list"][int(k)] for k in writingIndex]

            if f"intent_round_{self.intent_round}" not in self.multiturn_intent_rounds.keys():
                ###### """Initiate the 'conversation_intent_round_k' dictionary """ ######
                self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"] = {
                    "user_intent": None,
                    "binded": False,
                    "attributes": None
                }

            ###### Stage1: User Intent ######   
            if self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["binded"] is True:
                user_intent_check = self.nlu(explainInput)
                if user_intent_check != "none":
                    user_intent = user_intent_check
                    if user_intent_check != self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]['user_intent']:
                        self.intent_round += 1
                        self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"] = {
                            "user_intent": None,
                            "binded": False,
                            "attributes": None
                        }
                else:
                    user_intent = self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]['user_intent']
            else:
                user_intent = self.nlu(explainInput)




            ###### Stage2: Multi-turn Checks ######  
            if user_intent in ["confidence", "example", "attribution", "counterfactual"]:
                
                ###### Ensure the user select one sentence for explanations ######
                if len(writingInput) == 0:
                    response = "You are asking for <strong>instance-wise explanations</strong>, and <strong>please select (double click) one sentence</strong> to be explained."
                    return [response], response_indicator
                
                ###### Collect the variables to generate explanations. ###### 
                check_response = self.check_user_input_variables(user_intent, explainInput)


                responses = []
                for k in range(len(writingInput)):

                    ###### Stage3: Generating Explanations ######
                    explanations = self.explainer.generate_explanation(user_intent, writingInput[k], predictLabel[k], self.multiturn_intent_rounds["conference"], self.global_explanations_data, **self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"])

                    ###### Stage4: Responses ######
                    response = self.nlg(explanations)
                    responses.append(response)


                if check_response is not None:
                    if len(check_response) == 2:
                        return [check_response[1]], response_indicator
                    else:
                        return [responses, check_response], 6

                else:
                    self.intent_round += 1
                    return responses, response_indicator


            elif user_intent in ["meta-data", "meta-model", "quality-score", "aspect-distribution", "sentence-length"]:
            
                ###### Stage1: Generating Explanations ######
                explanations = self.explainer.generate_explanation(user_intent, "", "", self.multiturn_intent_rounds["conference"], self.global_explanations_data, **self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"])

                ###### Stage2: Responses ######
                response = self.nlg(explanations)

                self.intent_round += 1
                return [response], response_indicator

            else:
                explanations = "We currently can't answer this quesiton. Would you like to ask another questions?"
                response = self.nlg(explanations)
                self.intent_round += 1
                return [response], response_indicator














            # if self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]["binded"] is True:
            #     user_intent = self.multiturn_intent_rounds[f"intent_round_{self.intent_round}"]['user_intent']
            # else:
            #     user_intent = self.nlu(explainInput)




    # def writing_feedback(self, explainInput, writingInput, predictOutputs, inputTexts):

    #     ###### Info Collection Results ######
    #     response = f"Nice! I'm comparing your submission with <strong>{self.global_explanations_data[explainInput]['paper_count']}</strong> {explainInput} paper abstracts."

    #     ###### Overall Quality Score ######
    #     analysis_outputs = self._writing_analyzer(predictOutputs, explainInput, inputTexts)
    #     response_overall_score = f"<br><br><p class='overall-score'> Your <strong>Overall Quality Score</strong> = <strong>{analysis_outputs['abstract_score']:0.0f}</strong> (out of 5).</p> "
        
    #     ###### Instance-wise Score ######
    #     feedback_improvement = analysis_outputs['instance_results']
    #     if len(feedback_improvement) > 0:
    #         response_improvement = "<br>" + feedback_improvement
    #     else:
    #         response_improvement = "Your current writing looks good to me. Great job!"

    #     # return response + response_overall_score + response_improvement

    #     return response + response_improvement



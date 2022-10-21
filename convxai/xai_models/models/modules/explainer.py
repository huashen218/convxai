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
import torch.nn as nn
# import sys
# sys.path.append("/data/hua/workspace/projects/convxai/src/")


from .explainers import *
from .nlu import XAI_User_Intents
from convxai.writing_models.models import *



logger = logging.getLogger("my-logger")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT)
logger.setLevel(logging.INFO)



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')




class Model_Explainer(object):

    def __init__(self, configs):
        super(Model_Explainer, self).__init__()
        self.configs = configs
        logging.info("\nLoading writing models to be explained......")
        self.diversity_model  = DiversityModel(saved_model_dir=self.configs["scientific_writing"]["diversity_model_dir"])
        self.quality_model = QualityModel(saved_model_dir=self.configs["scientific_writing"]["quality_model_dir"])
        self.attribution_explainer = AttributionExplainer(configs)



    def generate_explanation(self, user_intent_detection, writingInput, predictLabel, conference, global_explanations_data, **kwargs):
        """writingInput: List[str]. A list of all input text.
                writingInput = ' '.join(writingInput)
        """

        ############ Global Explanations ############
        if user_intent_detection == XAI_User_Intents[0]:
            explanations = self.explain_meta_data(conference, global_explanations_data)

        elif user_intent_detection == XAI_User_Intents[1]:
            explanations = self.explain_meta_model()

        elif user_intent_detection == XAI_User_Intents[2]:
            explanations = self.explain_quality_score(conference, global_explanations_data)

        elif user_intent_detection == XAI_User_Intents[3]:
            explanations = self.explain_aspect_distribution(conference, global_explanations_data)

        elif user_intent_detection == XAI_User_Intents[8]:
            explanations = self.explain_sentence_length(conference, global_explanations_data)


        ############ Local Explanations ############
        elif user_intent_detection == XAI_User_Intents[4]:
            explanations = self.explain_confidence(writingInput, **kwargs)

        elif user_intent_detection == XAI_User_Intents[5]:
            explanations = self.explain_example(writingInput, predictLabel, conference, **kwargs)
        
        elif user_intent_detection == XAI_User_Intents[6]:
            explanations = self.explain_attribution(writingInput, **kwargs)

        elif user_intent_detection == XAI_User_Intents[7]:
            explanations = self.explain_counterfactual(writingInput, **kwargs)


        else:
            explanations = "We currently can't answer this quesiton. Would you like to ask another questions?"
            
        return explanations




    ############ Global Explanations ############
    def explain_meta_data(self, conference, global_explanations_data):
        """Global Explanations
        """
        explanations = f"""Sure! We are comparing your writing with our collected <strong>{conference} Paper Abstract</strong> dataset to generate the above review. The dataset includes <strong>{global_explanations_data[conference]['sentence_count']} sentences</strong> in <strong>{global_explanations_data[conference]['paper_count']} papers</strong>. 
        """
        return explanations


    def explain_meta_model(self):
        """Global Explanations
        <span style='color:#819CA3;'></span>
        """
        explanations = "Of course! The <strong>Writing Structure Model</strong> is a <a class='post-link' href='https://arxiv.org/pdf/1903.10676.pdf' target='_blank'>SciBERT</a> based classifier finetuned with the <a class='post-link' href='https://arxiv.org/pdf/2005.02367.pdf' target='_blank'>CODA-19</a> dataset. Also, the <strong>Writing Style Model</strong> is a <a class='post-link' href='https://openai.com/blog/tags/gpt-2/' target='_blank'>GPT-2</a> based generative model finetuned with <strong>9935 abstracts</strong> from <a class='post-link' href='https://dl.acm.org/conference/chi' target='_blank'>CHI</a>, <a class='post-link' href='https://aclanthology.org/venues/acl/' target='_blank'>ACL</a> and <a class='post-link' href='https://iclr.cc/Conferences/2023' target='_blank'>ICLR</a> papers (click the terms to view more)."
        return explanations


    def explain_quality_score(self, conference, global_explanations_data):
        """Global Explanations
        table generator: https://www.rapidtables.com/web/tools/html-table-generator.html
        """
        
        explanation1 = f"""
        We use each sentence's <strong>Perplexity</strong> value (predicted by the GPT-2 model) to derive the <strong>Quality Score</strong>. Lower perplexity means your writing is more similar to the {conference} papers.
        <br>
        We divide into five levels as below based on [20-th, 40-th, 60-th, 80-th] percentiles of the {conference} papers' perplexity scores (i.e., [{global_explanations_data[conference]['abstract_score_range'][0]}, {global_explanations_data[conference]['abstract_score_range'][1]}, {global_explanations_data[conference]['abstract_score_range'][3]}, {global_explanations_data[conference]['abstract_score_range'][4]}]).
        """

        explanation2 = """
        <br>
        <style>
            .demo {
                border:1px solid #EDEDED;
                border-collapse:separate;
                border-spacing:2px;
                padding:5px;
            }
            .demo th {
                border:1px solid #EDEDED;
                padding:5px;
                background:#D6D6D6;
            }
            .demo td {
                border:1px solid #EDEDED;
                text-align:center;
                padding:5px;
                background:#F5F5F5;
            }
        </style>
        """

        explanation3 = f"""
        <table class="demo">
            <caption><br></caption>
            <thead>
            <tr>
                <th>Quality Score</th>
                <th>Perplexity (PPL)</th>
            </tr>
            </thead>
            <tbody>
            <tr>
                <td>1 (lowest)</td>
                <td>{global_explanations_data[conference]['abstract_score_range'][4]} &lt; PPL<br></td>
            </tr>
            <tr>
                <td>2</td>
                <td>{global_explanations_data[conference]['abstract_score_range'][3]} &lt; PPL &lt;= {global_explanations_data[conference]['abstract_score_range'][4]}&nbsp;</td>
            </tr>
            <tr>
                <td>3</td>
                <td>{global_explanations_data[conference]['abstract_score_range'][1]} &lt; PPL &lt;= {global_explanations_data[conference]['abstract_score_range'][3]}&nbsp;</td>
            </tr>
            <tr>
                <td>4</td>
                <td>{global_explanations_data[conference]['abstract_score_range'][0]} &lt; PPL &lt;= {global_explanations_data[conference]['abstract_score_range'][1]}&nbsp;</td>
            </tr>
            <tr>
                <td>5 (highest)</td>
                <td>PPL &lt;= {global_explanations_data[conference]['abstract_score_range'][0]}&nbsp;</td>
            </tr>
            <tbody>
        </table>
        """
        return explanation1 + explanation2 + explanation3


    def explain_aspect_distribution(self, conference, global_explanations_data):
        """Global Explanations
        """
        explanation1 = """We use the Research Aspects Model to generate <strong>aspect sequences</strong> of all 9935 paper abstracts. Then we cluster these sequences into <strong>five patterns</strong> as below. We compare your writing with these patterns for review.
        <br>
        <style>
            .demo {
                border:1px solid #EDEDED;
                border-collapse:separate;
                border-spacing:2px;
                padding:5px;
            }
            .demo th {
                border:1px solid #EDEDED;
                padding:5px;
                background:#D6D6D6;
            }
            .demo td {
                border:1px solid #EDEDED;
                text-align:center;
                padding:5px;
                background:#F5F5F5;
            }
        </style>
        """

        explanation2 = f""" 
        <table class="demo">
            <caption><br></caption>
            <thead>
            <tr>
                <th>Types</th>
                <th>Patterns</th>
            </tr>
            </thead>
            <tbody>
            <tr>
                <td>Pattern1</td>
                <td>{global_explanations_data[f'Aspect_Patterns-{conference}'][0]}</td>
            </tr>
            <tr>
                <td>Pattern2&nbsp;</td>
                <td>{global_explanations_data[f'Aspect_Patterns-{conference}'][1]}</td>
            </tr>
            <tr>
                <td>Pattern3&nbsp;</td>
                <td>{global_explanations_data[f'Aspect_Patterns-{conference}'][2]}</td>
            </tr>
            <tr>
                <td>Pattern4</td>
                <td>{global_explanations_data[f'Aspect_Patterns-{conference}'][3]}</td>
            </tr>
            <tr>
                <td>Pattern5</td>
                <td>{global_explanations_data[f'Aspect_Patterns-{conference}'][4]}</td>
            </tr>
            <tbody>
        </table>
        """
        
        return explanation1 + explanation2


    def explain_sentence_length(self, conference, global_explanations_data):
        # explanation = f"""
        # The [20th, 40th, 50th, 60th, 80th] percentiles of the sentence lengths in the {conference} conference are <strong>{global_explanations_data[conference]["sentence_length"]}</strong> words. 
        # """
        explanation = f"""
        The [mean-2*std, mean-std, mean, mean+std, mean+2*std] percentiles of the sentence lengths in the {conference} conference are <strong>{global_explanations_data[conference]["sentence_length"]}</strong> words. 
        """
        return explanation



    ############ Local Explanations ############
    def explain_confidence(self, input, **kwargs):
        """Local Explanations: XAI Algorithm: provide the output confidence of the writing support models.
        """
        ######### Explaining #########
        predict, probability = self.diversity_model.generate_confidence(input)     # predict: [4];  probability [0.848671]
        
        ######### NLG #########
        label = diversity_model_label_mapping[predict[0]]
        nlg_template = f"Given your selected sentence = <span class='text-info'>{input}</span>, the model predicts a <strong>'{label}' aspect</strong> label with <strong>confidence score = {probability[0]:.4f}</strong>. "
        response = nlg_template
        return response



    def explain_example(self, input, predictLabel, conference, **kwargs):
        """XAI Algorithm - Confidence: 
            Paper: An Empirical Comparison of Instance Attribution Methods for NLP (https://aclanthology.org/2021.naacl-main.75.pdf)
        """
        ######### User Input Variable #########
        top_k = kwargs["attributes"]["top_k"] if kwargs["attributes"]["top_k"] is not None else 3

        ######### Explaining #########
        self.example_explainer = ExampleExplainer(conference)
        embeddings = self.diversity_model.generate_embeddings(input)

        if kwargs["attributes"]["aspect"] is not None:
            label = label_mapping[kwargs["attributes"]["aspect"]]
        else:
            label = predictLabel

        filter_index = np.where(self.example_explainer.diversity_aspect_list_tmp == label)[0]
        self.example_explainer.diversity_x_train_embeddings_tmp = np.array(self.example_explainer.diversity_x_train_embeddings_tmp)[filter_index]
        similarity_scores = np.matmul(embeddings, np.transpose(self.example_explainer.diversity_x_train_embeddings_tmp, (1,0)))[0]    ###### train_emb = torch.Size(137171, 768)  

        if kwargs["attributes"]["aspect"] is not None:
            final_top_index = filter_index[np.argsort(similarity_scores)[::-1][:top_k]]
        else:
            final_top_index =  np.argsort(similarity_scores)[::-1][:top_k]

        top_text = np.array(self.example_explainer.diversity_x_train_text_tmp)
        top_title = np.array(self.example_explainer.diversity_x_train_title_tmp)
        top_link = np.array(self.example_explainer.diversity_x_train_link_tmp)
        top_aspect = np.array(self.example_explainer.diversity_aspect_list_tmp)

        nlg_template = f"The top-{top_k} similar examples (i.e., of selected-sentence = '<i><span class='text-info'>{input}</span>') from the <strong>{conference}</strong> dataset are (Conditioned on <strong>label={diversity_model_label_mapping[label]}</strong>):"

        for t in final_top_index:
            nlg_template += f"<br> <strong>sample-{t+1}</strong> - <a class='post-link' href='{top_link[t].decode('UTF-8')}' target='_blank'>{top_text[t].decode('UTF-8')}</a>."
        response = nlg_template
        return response



    def _attribution_visualize(self, all_predic_toks, important_indices):
        text = " "
        for k in range(len(all_predic_toks)):
            if k in important_indices:
                text += ' <b><span style="font-weight: bold; background-color: #F9B261; border-radius: 5px;">' + all_predic_toks[k] + '</span></b>'
            else:
                text += ' <b><span style="font-weight: normal;">' + all_predic_toks[k] + '</span></b>'
        return text

        


    def explain_attribution(self, input, **kwargs):
        """XAI Algorithm - Attribution: 
            Implementation Reference: https://stackoverflow.com/questions/67142267/gradient-based-saliency-of-input-words-in-a-pytorch-model-from-transformers-libr
        """
        top_k = kwargs["attributes"]["top_k"] if kwargs["attributes"]["top_k"] is not None else 3
        all_predic_toks, ordered_predic_tok_indices = self.attribution_explainer.get_sorted_important_tokens_nonordered(input)
        important_indices = ordered_predic_tok_indices[:top_k]
        nlg_template = f"The <strong>TOP-{top_k}</strong> important words are highlighted as below: <br><br>"
        html_text = self._attribution_visualize(all_predic_toks, important_indices)
        response = nlg_template + html_text
        return response



    def explain_counterfactual(self, input, **kwargs):
        """XAI Algorithm #6: MICE
        Reference paper: Explaining NLP Models via Minimal Contrastive Editing (MICE)
        """
        contrast_label_idx_input = label_mapping[kwargs["attributes"]["contrast_label"]] if kwargs["attributes"]["contrast_label"] is not None else -2
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
            nlg_template = "The most likely counterfactual label is <strong>'{}'</strong>. You can get this label by revising from \n'<span class='text-info'>{}</span>\n' into: \n <br>'<em><span class='text-secondary'>{}</span></em>'. <br>I'm confident with this revision with <strong>confidence score={:.4f}</strong>."
            response = nlg_template.format(output['counterfactual_label'], output['original_input'], counterfactual_output, output['counterfactual_confidence'])
        else:
            response = f"Sorry that I currently can't find counterfactual examples for this sentence, please try other explanations on it : )"
        return response




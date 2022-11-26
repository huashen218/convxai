#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code supports the web server of the ConvXAI system.
# Copyright (c) Hua Shen 2022.
#

import torch.nn as nn
from .nlu import XAI_User_Intents

class XAI_NLG_Module(nn.Module):

    def __init__(self):
        super(XAI_NLG_Module, self).__init__()
        """Template-based conversational XAI generation.
        """


    def forward(self, user_intent_detection, explanation_dict):
        r"""Template-based NLG: The XAI_NLG module aims to generate the free-text responses to the users based on the explanations.
        
        Args:
            user_intent_detection (`str`): The user intent.
            explanation (Dict[`str`, `str`]): The dictionary of generated explanations.

        Returns:
            (`str`): The detected user intent.
        """

        ############ Global Explanations ############
        if user_intent_detection == XAI_User_Intents[0]:
            response = self.template_meta_data(**explanation_dict)

        elif user_intent_detection == XAI_User_Intents[1]:
            response = self.template_meta_model()

        elif user_intent_detection == XAI_User_Intents[2]:
            response = self.template_quality_score(**explanation_dict)

        elif user_intent_detection == XAI_User_Intents[3]:
            response = self.template_aspect_distribution(**explanation_dict)

        elif user_intent_detection == XAI_User_Intents[8]:
            response = self.template_sentence_length(**explanation_dict)

        ############ Local Explanations ############
        elif user_intent_detection == XAI_User_Intents[4]:
            response = self.template_confidence(**explanation_dict)

        elif user_intent_detection == XAI_User_Intents[5]:
            response = self.template_example(**explanation_dict)
        
        elif user_intent_detection == XAI_User_Intents[6]:
            response = self.template_attribution(**explanation_dict)

        elif user_intent_detection == XAI_User_Intents[7]:
            response = self.template_counterfactual(**explanation_dict)
        else:
            response = "We currently can't answer this quesiton. Would you like to ask another questions?"
            
        return response



    ############ Global Explanations ############
    def template_meta_data(self, conference, global_explanations_data):
        """Global Explanations
        """
        response = f"""Sure! We are comparing your writing with our collected <strong>{conference} Paper Abstract</strong> dataset to generate the above review. The dataset includes <strong>{global_explanations_data[conference]['sentence_count']} sentences</strong> in <strong>{global_explanations_data[conference]['paper_count']} papers</strong>. 
        """
        return response


    def template_meta_model(self):
        """Global Explanations
        <span style='color:#819CA3;'></span>
        """
        response = "Of course! The <strong>Writing Structure Model</strong> is a <a class='post-link' href='https://arxiv.org/pdf/1903.10676.pdf' target='_blank'>SciBERT</a> based classifier finetuned with the <a class='post-link' href='https://arxiv.org/pdf/2005.02367.pdf' target='_blank'>CODA-19</a> dataset. Also, the <strong>Writing Style Model</strong> is a <a class='post-link' href='https://openai.com/blog/tags/gpt-2/' target='_blank'>GPT-2</a> based generative model finetuned with <strong>9935 abstracts</strong> from <a class='post-link' href='https://dl.acm.org/conference/chi' target='_blank'>CHI</a>, <a class='post-link' href='https://aclanthology.org/venues/acl/' target='_blank'>ACL</a> and <a class='post-link' href='https://iclr.cc/Conferences/2023' target='_blank'>ICLR</a> papers (click the terms to view more)."
        return response


    def template_quality_score(self, conference, global_explanations_data):
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
        response = explanation1 + explanation2 + explanation3
        return response


    def template_aspect_distribution(self, conference, global_explanations_data):
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
        response = explanation1 + explanation2
        return response


    def template_sentence_length(self, conference, global_explanations_data):
        response = f"""
        The [mean-2*std, mean-std, mean, mean+std, mean+2*std] percentiles of the sentence lengths in the {conference} conference are <strong>{global_explanations_data[conference]["sentence_length"]}</strong> words. 
        """
        return response



    ############ Local Explanations ############
    def template_confidence(self, input, label, probability):
        """Local Explanations: XAI Algorithm: provide the output confidence of the writing support models.
        """
        response = f"Given your selected sentence = <span class='text-info'>{input}</span>, the model predicts a <strong>'{label}' aspect</strong> label with <strong>confidence score = {probability[0]:.4f}</strong>. "
        return response


    def template_example(self, top_k, input, conference, label, final_top_index, top_link, top_text):
        """XAI Algorithm - Confidence: 
            Paper: An Empirical Comparison of Instance Attribution Methods for NLP (https://aclanthology.org/2021.naacl-main.75.pdf)
        """
        response = f"The top-{top_k} similar examples (i.e., of selected-sentence = '<i><span class='text-info'>{input}</span>') from the <strong>{conference}</strong> dataset are (Conditioned on <strong>label={label}</strong>):"

        for t in final_top_index:
            response += f"<br> <strong>sample-{t+1}</strong> - <a class='post-link' href='{top_link[t].decode('UTF-8')}' target='_blank'>{top_text[t].decode('UTF-8')}</a>."
        return response



    def _attribution_visualize(self, all_predic_toks, important_indices):
        text = " "
        for k in range(len(all_predic_toks)):
            if k in important_indices:
                text += ' <b><span style="font-weight: bold; background-color: #F9B261; border-radius: 5px;">' + all_predic_toks[k] + '</span></b>'
            else:
                text += ' <b><span style="font-weight: normal;">' + all_predic_toks[k] + '</span></b>'
        return text

        


    def template_attribution(self, top_k, all_predic_toks, important_indices):
        """XAI Algorithm - Attribution: 
            Implementation Reference: https://stackoverflow.com/questions/67142267/gradient-based-saliency-of-input-words-in-a-pytorch-model-from-transformers-libr
        """
        nlg_template = f"The <strong>TOP-{top_k}</strong> important words are highlighted as below: <br><br>"
        html_text = self._attribution_visualize(all_predic_toks, important_indices)
        response = nlg_template + html_text
        return response



    def template_counterfactual(self, output, counterfactual_output, counterfactual_exists=True):
        """XAI Algorithm #6: MICE
        Reference paper: Explaining NLP Models via Minimal Contrastive Editing (MICE)
        """
        if counterfactual_exists:
            nlg_template = "The most likely counterfactual label is <strong>'{}'</strong>. You can get this label by revising from \n'<span class='text-info'>{}</span>\n' into: \n <br>'<em><span class='text-secondary'>{}</span></em>'. <br>I'm confident with this revision with <strong>confidence score={:.4f}</strong>."
            response = nlg_template.format(output['counterfactual_label'], output['original_input'], counterfactual_output, output['counterfactual_confidence'])
        else:
            response = f"Sorry that I currently can't find counterfactual examples for this sentence, please try other explanations on it : )"
        return response

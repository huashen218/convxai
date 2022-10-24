from transformers import T5Tokenizer, T5Model, T5Config
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from allennlp.predictors import Predictor, TextClassifierPredictor
from allennlp_models.classification import StanfordSentimentTreeBankDatasetReader
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

import torch
import os
import csv
import heapq
import sys
import operator
from tqdm import tqdm
import re
import nltk
import warnings
import argparse
import pandas as pd
import numpy as np
import random
import time
import logging
import json

# Local imports
from convxai.xai_models.trainers.explainers.counterfactual_explainer.src.utils import *
from convxai.xai_models.trainers.explainers.counterfactual_explainer.src.edit_finder import EditFinder, EditEvaluator, EditList
from convxai.xai_models.trainers.explainers.counterfactual_explainer.src.editor import Editor, RaceEditor
from convxai.xai_models.trainers.explainers.counterfactual_explainer.src.masker import Masker, RandomMasker, GradientMasker

logger = logging.getLogger("my-logger")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT)
logger.setLevel(logging.INFO)


def get_grad_sign_direction(grad_type, grad_pred):
    """ Helper function to get sign direction. When grad_type is signed, 
    determines whether to get most negative or positive gradient values. 
    This should depend on grad_pred, i.e. what label is being used to 
    compute gradients for masking. 
    
    During Stage Two, we want to mask tokens that push *away* from the contrast 
    label or *towards* the original label. 
    
    Sign direction plays no role if only gradient *magnitudes* are used 
        (i.e. if grad_type is not signed, but involves taking the l1/l2 norm.)
    """
    assert grad_pred in ["contrast", "original"]
    assert grad_type in ["integrated_l1", "integrated_signed", "normal_l1", 
            "normal_signed", "normal_l2", "integrated_l2"]
    if "signed" in grad_type and "contrast" in grad_pred:
        sign_direction = -1
    elif "signed" in grad_type and "original" in grad_pred:
        sign_direction = 1
    else:
        sign_direction = None
    return sign_direction

def load_editor_weights(editor_model, editor_path):
    """ Loads Editor weights from editor_path """
    if os.path.isdir(editor_path):
        editor_path = os.path.join(editor_path, "best.pth")
        if not os.path.exists(editor_path):
            raise NotImplementedError(f"If directory given for editor_path, \
                    it must contain a 'best.pth' file but found none in given \
                    dir. Please give direct path to file containing weights.")

    logger.info(f"Loading Editor weights from: {editor_path}")
    editor_model.load_state_dict(torch.load(editor_path))
    return editor_model

def load_models(args):
    """ Loads Predictor and Editor by task and other args """

    logger.info("Loading models...")
    predictor = load_diversity_model()

    editor_tokenizer_wrapper = PretrainedTransformerTokenizer(
            't5-base', max_length=args.model.model_max_length)
    editor_tokenizer, editor_model = load_base_t5(
                       max_length=args.model.model_max_length)
    device = get_device()
    editor_model = load_editor_weights(editor_model, args.meta.editor_path)
    editor_model = editor_model.to(device)
    
    sign_direction = get_grad_sign_direction(
            args.mask.grad_type, args.misc.grad_pred) 
    
    masker = GradientMasker(args.search.max_mask_frac, 
            editor_tokenizer_wrapper, predictor, 
            args.model.model_max_length,
            grad_type=args.mask.grad_type, 
            sign_direction=sign_direction)

    if "race" in args.meta.task:
        editor = RaceEditor(editor_tokenizer_wrapper, editor_tokenizer, 
                editor_model, masker, 
                num_gens=args.generation.num_generations, 
                num_beams=args.generation.generation_num_beams, 
                grad_pred=args.misc.grad_pred, 
                generate_type=args.generation.generate_type, 
                length_penalty=args.generation.length_penalty, 
                no_repeat_ngram_size=args.generation.no_repeat_ngram_size, 
                top_p=args.generation.top_p,
                top_k=args.generation.top_k, 
                verbose=False, 
                editable_key="article")
    else:
        editor = Editor(editor_tokenizer_wrapper, editor_tokenizer, 
                editor_model, masker, 
                num_gens=args.generation.num_generations, 
                num_beams=args.generation.generation_num_beams, 
                grad_pred=args.misc.grad_pred, 
                generate_type=args.generation.generate_type, 
                no_repeat_ngram_size=args.generation.no_repeat_ngram_size, 
                top_p=args.generation.top_p, 
                top_k=args.generation.top_k, 
                length_penalty=args.generation.length_penalty, 
                verbose=False, dataset_reader = get_dataset_reader(args.meta.task, predictor))
                        
    logger.info("Done loading models.")
    return editor, predictor



###### Main Function ###### 
def run_edit_test(args):
    """ Runs Stage 2 on test inputs by task. """

    task_dir = os.path.join(args.meta.results_dir, args.meta.task)
    stage_two_dir = os.path.join(task_dir, f"edits/{args.meta.stage2_exp}")
   
    if not os.path.exists(stage_two_dir):
        os.makedirs(stage_two_dir)

    logger.info(f"Task dir: {task_dir}")
    logger.info(f"Stage two dir: {stage_two_dir}")

    # Save args
    args_path = os.path.join(stage_two_dir, "stage_two_args.json")
    write_args(args_path, args)
    
    out_file = os.path.join(stage_two_dir, "edits.csv")
    meta_log_file = os.path.join(stage_two_dir, "meta_log.txt")

    meta_f = open(meta_log_file, 'w', 1)

    # Load models and Edit objects 
    editor, predictor = load_models(args)
    dr = get_dataset_reader(args.meta.task, predictor)
    
    edit_evaluator = EditEvaluator()
    edit_finder = EditFinder(predictor, editor, 
            beam_width=args.search.beam_width, 
            max_mask_frac=args.search.max_mask_frac,
            search_method=args.search.search_method,
            max_search_levels=args.search.max_search_levels)

    # Get inputs
    # inputs = dr.get_inputs('test')
    # strings, labels = dr.get_inputs("/home/hqs5468/hua/workspace/projects/convxai/src/convxai/writing_models/data/diversity_dataset/coda19/CODA19_v1_20200504/human_label", return_labels=True)
    inputs, labels = dr.get_inputs("/home/hqs5468/hua/workspace/projects/convxai/src/convxai/writing_models/data/diversity_dataset/coda19/CODA19_v1_20200504/human_label", phrase="test", return_labels=True)
    
    ######## Inputs ########
    # ===>>>inputs ['Dengue has a negative impact in low-and lower middle-income countries ,', 'but also affects upper middle-and high-income countries .', 'Despite the efforts at controlling this disease ,', 'it is unclear why dengue remains an issue in affluent countries .', 'A better understanding of dengue
    #  epidemiology and its burden ,']
    # ====>>>>>labels ['background', 'background', 'background', 'background', 'background']
    ########################


    np.random.seed(0)
    input_indices = np.array(range(len(inputs)))
    np.random.shuffle(input_indices)


    # Find edits and write to file
    with open(out_file, "w") as csv_file:
        fieldnames = ["data_idx", "sorted_idx", "orig_pred", "new_pred", 
                "contrast_pred", "orig_contrast_prob_pred", 
                "new_contrast_prob_pred", "orig_input", "edited_input", 
                "orig_editable_seg", "edited_editable_seg", 
                "minimality", "num_edit_rounds", "mask_frac", 
                "duration", "error"]
        writer = csv.writer(csv_file, delimiter="\t")
        writer.writerow(fieldnames)



        for idx, i in tqdm(enumerate(input_indices), total=len(input_indices)):
            inp = inputs[i]
            logger.info(wrap_text(f"ORIGINAL INSTANCE ({i}): {inp}"))

            start_time = time.time()
            error = False

            try:
                edited_list = edit_finder.minimally_edit(inp, 
                        max_edit_rounds=args.search.max_edit_rounds, 
                        edit_evaluator=edit_evaluator, max_length = int(args.model.model_max_length))

                torch.cuda.empty_cache()
                sorted_list = edited_list.get_sorted_edits() 
                """
                {'edited_editable_seg': ' AV. DMVs , however , lack conventional endoplasmic reticulum ( ER ) or Golgi protein markers ,', 
                'edited_input': ' AV. DMVs , however , lack conventional endoplasmic reticulum ( ER ) or Golgi protein markers ,', 
                'minimality': 0.15789473684210525, 
                'masked_sentence': '<extra_id_0>V<extra_id_1> DMVs , however , lack conventional endoplasmic reticulum ( ER ) or Golgi protein markers ,', 
                'edited_contrast_prob': 0.3411323, 
                'edited_label': 'background', 

                'mask_frac': 0.06875, 
                'num_edit_rounds': 1}
                """

                output = {
                    "original_input": inputs[i],
                    "label": labels[i],
                    "counterfactual_input": sorted_list[0]['edited_editable_seg'],
                    "counterfactual_label": sorted_list[0]['edited_label'],
                    "counterfactual_confidence": sorted_list[0]['edited_contrast_prob'],
                }


            except Exception as e:
                logger.info("ERROR: ", e)
                error = True
                sorted_list = []

            end_time = time.time()


            duration = end_time - start_time
            for s_idx, s in enumerate(sorted_list):
                writer.writerow([i, s_idx, edited_list.orig_label, 
                    s['edited_label'], edited_list.contrast_label, 
                    edited_list.orig_contrast_prob, s['edited_contrast_prob'], 
                    edited_list.orig_input, s['edited_input'], 
                    edited_list.orig_editable_seg, 
                    s['edited_editable_seg'], s['minimality'], 
                    s['num_edit_rounds'], s['mask_frac'], duration, error])   ### 8.4481
                csv_file.flush()
            if sorted_list == []:
                writer.writerow([i, 0, edited_list.orig_label, 
                    None, edited_list.contrast_label, 
                    edited_list.orig_contrast_prob, None, 
                    edited_list.orig_input, None, 
                    edited_list.orig_editable_seg, 
                    None, None, None, None, duration, error]) 
                csv_file.flush()
                meta_f.flush()

    csv_file.close()
    meta_f.close()



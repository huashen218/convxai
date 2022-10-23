import torch
from convxai.xai_models.trainers.explainers.counterfactual_explainer.src.utils import *
from convxai.xai_models.trainers.explainers.counterfactual_explainer.src.edit_finder import EditFinder, EditEvaluator, EditList
from convxai.xai_models.trainers.explainers.counterfactual_explainer.src.stage_two import load_models


class CounterfactualExplainer(object):
    
    def __init__(self):
        super().__init__()
        self.args = get_args("stage2")
        self.editor, self.predictor = load_models(self.args)
        self.edit_evaluator = EditEvaluator()
        self.edit_finder = EditFinder(self.predictor, self.editor, 
                beam_width=self.args.search.beam_width, 
                max_mask_frac=self.args.search.max_mask_frac,
                search_method=self.args.search.search_method,
                max_search_levels=self.args.search.max_search_levels)


    def generate_counterfactual(self, input_text, contrast_label):
        edited_list = self.edit_finder.minimally_edit(input_text, 
                                                        contrast_pred_idx_input = contrast_label, ### contrast_pred_idx specifies which label to use as the contrast. Defaults to -2, i.e. use label with 2nd highest pred prob.
                                                        max_edit_rounds=self.args.search.max_edit_rounds, 
                                                        edit_evaluator=self.edit_evaluator, 
                                                        max_length = int(self.args.model.model_max_length))

        torch.cuda.empty_cache()
        sorted_list = edited_list.get_sorted_edits() 

        if len(sorted_list) > 0:
            output = {
                "original_input": input_text,
                "counterfactual_input": sorted_list[0]['edited_editable_seg'],
                "counterfactual_label": sorted_list[0]['edited_label'],
                "counterfactual_confidence": sorted_list[0]['edited_contrast_prob'],
            }
        else:
            output = []
        return output
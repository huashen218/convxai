# Local imports
from convxai.xai_models.trainers.explainers.counterfactual_explainer.src.stage_two import run_edit_test
from convxai.xai_models.trainers.explainers.counterfactual_explainer.src.utils import get_args

if __name__ == '__main__':
    args = get_args("stage2")
    run_edit_test(args)

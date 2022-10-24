# Local imports

from convxai.xai_models.trainers.explainers.counterfactual_explainer.src.stage_one import run_train_editor
from convxai.xai_models.trainers.explainers.counterfactual_explainer.src.utils import get_args, load_diversity_model, get_dataset_reader


if __name__ == '__main__':

    args = get_args("stage1")
    predictor = load_diversity_model()
    dr = get_dataset_reader(args.meta.task, predictor)
    run_train_editor(predictor, dr, args)
    

from ..ai_models import *


############ Local Explanations ############
def explain_confidence(model, input, **kwargs):
    """Local Explanations: XAI Algorithm: provide the output confidence of the writing support models.
    """
    ######### Explaining #########
    predict, probability = model.generate_confidence(input)     # predict: [4];  probability [0.848671]
    
    ######### NLG #########
    label = diversity_model_label_mapping[predict[0]]
    nlg_template = f"Given your selected sentence = <span class='text-info'>{input}</span>, the model predicts a <strong>'{label}' aspect</strong> label with <strong>confidence score = {probability[0]:.4f}</strong>. "
    response = nlg_template
    return response
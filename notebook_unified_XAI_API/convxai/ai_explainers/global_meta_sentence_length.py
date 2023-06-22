
def explain_sentence_length(conference, global_explanations_data):
    explanation = f"""
    The [20th, 40th, 50th, 60th, 80th] percentiles of the sentence lengths in the {conference} conference are <strong>{global_explanations_data[conference]["sentence_length"]}</strong> words. 
    """
    return explanation
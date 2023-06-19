
def explain_meta_data(conference, global_explanations_data):
    """Global Explanations
    """
    explanations = f"""Sure! We are comparing your writing with our collected <strong>{conference} Paper Abstract</strong> dataset to generate the above review. The dataset includes <strong>{global_explanations_data[conference]['sentence_count']} sentences</strong> in <strong>{global_explanations_data[conference]['paper_count']} papers</strong>. 
    """
    return explanations

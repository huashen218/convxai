

import sys
sys.path.append('/data/hua/workspace/projects/convxai/src/')

from convxai.writing_models.models import *
model_configs_dir = "/data/hua/workspace/projects/convxai/src/convxai/xai_models/models/model_configs.json"

def main():
    with open(model_configs_dir, 'r') as f:
        model_configs = json.load(f)
    diversity_model  = DiversityModel(model_configs["writing_model_dir"]["diversity_model"])
    quality_model = QualityModel(saved_model_dir=model_configs["writing_model_dir"]["quality_model"])


    diversity_model_embeddings_h5dir = model_configs["xai_emample_embeddings_dir"]["diversity_model"]
    quality_model_embeddings_h5dir = model_configs["xai_emample_embeddings_dir"]["quality_model"]
    

    with h5py.File(diversity_model_embeddings_h5dir, 'r') as infile:
        diversity_x_train_embeddings_tmp = np.empty(infile["x_train_embeddings"].shape, dtype=infile["x_train_embeddings"].dtype)
        infile["x_train_embeddings"].read_direct(diversity_x_train_embeddings_tmp)
        print(">>>diversity_x_train_embeddings_tmp:", diversity_x_train_embeddings_tmp.shape)

        diversity_x_train_text_tmp = np.empty(infile["x_train_text"].shape, dtype=infile["x_train_text"].dtype)
        infile["x_train_text"].read_direct(diversity_x_train_text_tmp)
        print(">>>diversity_x_train_embeddings_tmp:", diversity_x_train_embeddings_tmp.shape)


    with h5py.File(quality_model_embeddings_h5dir, 'r') as infile:
        quality_x_train_embeddings_tmp = np.empty(infile["x_train_embeddings"].shape, dtype=infile["x_train_embeddings"].dtype)
        infile["x_train_embeddings"].read_direct(quality_x_train_embeddings_tmp)   ### (1707, 768)
        # print(">>>quality_x_train_embeddings_tmp:", quality_x_train_embeddings_tmp.shape)  

        quality_x_train_text_tmp = np.empty(infile["x_train_text"].shape, dtype=infile["x_train_text"].dtype)
        infile["x_train_text"].read_direct(quality_x_train_text_tmp)




    writingInput = "Natural language understanding comprises a wide range of diverse tasks such as textual entailment ,question answering , semantic similarity assessment ,and document classification .Although large unlabeled text corpora are abundant ,labeled data for learning these specific tasks is scarce ,making it challenging for discriminatively trained models to perform adequately .We demonstrate that large gains on these tasks can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text ,followed by discriminative fine-tuning on each specific task ."



    ### examples
    k = 1
    quality_generate_embeddings = quality_model.generate_embeddings(writingInput)[-1]   ### (768)
    quality_similarity_scores = np.dot(quality_x_train_embeddings_tmp, quality_generate_embeddings)
    top_index =  np.argsort(quality_similarity_scores)[-k]
    top_1_text = quality_x_train_text_tmp[top_index]
    
    print("==>>>quality similarity_scores:", diversity_similarity_scores)
    print("quality similarity_scores shape", diversity_similarity_scores.shape)


    diversity_generate_embeddings = diversity_model.generate_embeddings(writingInput)
    diversity_similarity_scores = np.dot(diversity_x_train_embeddings_tmp, diversity_generate_embeddings)
    top_index =  np.argsort(diversity_similarity_scores)[-k]
    top_1_text = diversity_x_train_text_tmp[top_index]
    
    print("==>>>diversity similarity_scores:", diversity_similarity_scores)
    print("diversity similarity_scores shape", diversity_similarity_scores.shape)
    exit()


    # generate_confidence = diversity_model.generate_confidence(writingInput)





### Debug
if __name__ == '__main__':
    main()



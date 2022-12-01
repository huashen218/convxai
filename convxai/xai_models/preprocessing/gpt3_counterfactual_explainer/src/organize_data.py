import pandas as pd
from pathlib import Path
from transformers import GPT2Tokenizer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import joblib
import json

data_folder = "/home/appleternity/workspace/convxai/convxai/xai_models/preprocessing/global_xai_statistics"

def load_data():
    all_data = []
    for conference in ["ACL", "CHI", "ICLR"]:
        file_path = Path(data_folder, f"{conference}.csv")
        data = pd.read_csv(file_path)
        data = data.drop(["aspect_confidence", "perplexity"], axis="columns")
        data["conference"] = conference
        all_data.append(data)
    data = pd.concat(all_data).to_dict('records')
    return data

def organize_data():
    data = load_data()
    data = data[:]

    # (1) update "token_count" using GPT2/GPT3 tokenization
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    for sample in tqdm(data, "Tokenizing"):
        input_ids = tokenizer(sample["text"])['input_ids']
        sample["token_count"] = len(input_ids)

    # (2) add 
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.max_seq_length = 256
    text_list = [sample["text"] for sample in data]
    embeddings = model.encode(text_list, convert_to_tensor=True, show_progress_bar=True)
    embeddings = embeddings.cpu()
    
    # (3) save
    joblib.dump(embeddings, Path(data_folder, "embeddings_all-MiniLM-L6-v2.joblib"))
    with open(Path(data_folder, "text_info.json"), 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=2)


def main():
    organize_data()

if __name__ == "__main__":
    main()
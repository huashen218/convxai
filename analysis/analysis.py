import pandas as pd
from pathlib import Path
from nltk import word_tokenize
import string
import editdistance
from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance
import textstat
from sacremoses import MosesDetokenizer

def load_data():
    filename = "data/Writing-Artifacts-Evaluation.xlsx"
    data = pd.read_excel(filename)
    data = data.fillna("")

    data_list = []
    data_dict = {}
    for index, row in data.iterrows():
        sample = row.to_dict()
        
        if sample["Users"] == "":
            sample["Users"] = data_list[-1]["Users"]
            sample["System"] = data_list[-1]["System"]
            sample["Paper Link"] = data_list[-1]["Paper Link"]
            sample["Domain"] = data_list[-1]["Domain"]

        data_list.append(sample)
        data_dict[(sample["Users"], sample["System"], sample["Indicator"])] = sample

    return data_dict

def clean_text(text, remove_punctuation=False, lower=False):
    return [
        token if not lower else token.lower()
        for token in word_tokenize(text)
        if not remove_punctuation or token not in string.punctuation
    ]

def analyze_edit_distance():
    data = load_data()
    # key = (user_id, "ConvXAI", "Original Abstract")
    # key = (user_id, "ConvXAI", "Improved Abstract")
    # key = (user_id, "SelectXAI", "Original Abstract")
    # key = (user_id, "SelectXAI", "Improved Abstract")
    
    result = []
    for user_id in ["P1", "P2", "P3", "P4", "P5"]:
        for system in ["ConvXAI", "SelectXAI"]:
        
            key = (user_id, system, "Original Abstract")
            original = clean_text(data[key]["Abstracts"], remove_punctuation=True, lower=True)

            key = (user_id, system, "Improved Abstract")
            improved = clean_text(data[key]["Abstracts"], remove_punctuation=True, lower=True)

            print(original)
            print(improved)

            edit_distance = damerau_levenshtein_distance(original, improved)
            print(edit_distance, len(original), len(improved))

            normalized_edit_distance = normalized_damerau_levenshtein_distance(original, improved)
            print(normalized_edit_distance, len(original), len(improved))

            result.append({
                "user_id": user_id,
                "system": system,
                "original-length": len(original),
                "improved-length": len(improved),
                "edit-distance": edit_distance,
                "normalized-edit-distance": normalized_edit_distance,
            })

    table = pd.DataFrame(result)
    table = table.sort_values(by="system", axis="index")
    table.to_excel("data/edit-distance.xlsx")

def analyze_readability():
    data = load_data()
    detokenizer = MosesDetokenizer(lang='en')
    
    result = []
    for user_id in ["P1", "P2", "P3", "P4", "P5"]:
        for system in ["ConvXAI", "SelectXAI"]:
            for phase in ["Original Abstract", "Improved Abstract"]:
                key = (user_id, system, phase)
                text = detokenizer.detokenize(clean_text(data[key]["Abstracts"]))

                print(text)

                score = textstat.flesch_reading_ease(text)

                result.append({
                    "user_id": user_id,
                    "system": system,
                    "phase": phase,
                    "flesch_reading_ease": score,
                })

    table = pd.DataFrame(result)
    table = table.sort_values(by=["system", "phase"], axis="index")
    table.to_excel("data/readability.xlsx")

def main():
    # analyze_edit_distance()
    analyze_readability()

if __name__ == "__main__":
    main()
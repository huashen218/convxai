import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

import typer



DATA_FOLDER="../data"


def load_data(
    filename: Path = Path(DATA_FOLDER, "xai_intent_all.csv")
):
    table = pd.read_csv(filename, index_col=False)
    return table

def split_dataframe(
    data: pd.DataFrame,
    random_seed: int = 42,
    train_split: float = 0.7,
    valid_split: float = 0.1,
    test_split: float = 0.2,
):
    """
    split data in to train/valid/test set using the given ratio
    Make sure the split is stratified by label
    """
    rng = np.random.default_rng(random_seed)

    # get all unique labels
    label_list = data["label"].unique().tolist()
    label_list.sort()

    # split data by label
    split_data_list = []
    for label in label_list:
        label_data = data[data["label"]==label]
        label_data = label_data.reset_index(drop=True)

        # split
        random_index_list = rng.permutation(len(label_data))
        valid_num, test_num = int(len(label_data)*valid_split), int(len(label_data)*test_split)
        valid_index_list = random_index_list[0:valid_num]
        test_index_list = random_index_list[valid_num:valid_num+test_num]
        train_index_list = random_index_list[valid_num+test_num:]

        # split data
        label_train_data = label_data.iloc[train_index_list]
        label_valid_data = label_data.iloc[valid_index_list]
        label_test_data = label_data.iloc[test_index_list]

        # add split column
        label_train_data["split"] = "train"
        label_valid_data["split"] = "valid"
        label_test_data["split"] = "test"

        # append
        split_data_list.append(label_train_data)
        split_data_list.append(label_valid_data)
        split_data_list.append(label_test_data)

    # concat
    split_data = pd.concat(split_data_list, axis=0)
    return split_data

def process_data(
    input_file: Path = typer.Option(..., help="input file path"),
    output_folder: Path = typer.Option(..., help="output folder path"),
):
    table = load_data(input_file)
    table = split_dataframe(table)
    
    # create output_folder if not exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # save data into three json files (train/valid/test)
    # each sample is a dict with two keys: "text" and "label"
    for split in ["train", "valid", "test"]:
        split_data = table[table["split"]==split]

        # rename [sentence, label] into [text, label]
        split_data = split_data.rename(columns={"question": "text"})
        split_data = split_data[["text", "label"]]
        split_data = split_data.to_dict(orient="records")

        print(f"There are a total of {len(split_data)} {split} samples")

        # save
        filename = Path(output_folder, f"{split}.json")
        with open(filename, 'w', encoding='utf-8') as outfile:
            json.dump(split_data, outfile, indent=2)

if __name__ == "__main__":
    typer.run(process_data)



import logging
import h5py
import random
import numpy as np
from itertools import chain

from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import (
    default_data_collator,
)

logger = logging.getLogger(__name__)


def h5_load(filename, data_list, dtype=None, verbose=False):
    with h5py.File(filename, 'r') as infile:
        data = []
        for data_name in data_list:
            if dtype is not None:
                temp = np.empty(infile[data_name].shape, dtype=dtype)
            else:
                temp = np.empty(infile[data_name].shape, dtype=infile[data_name].dtype)
            infile[data_name].read_direct(temp)
            data.append(temp)
          
        if verbose:
            print("\n".join(
                "{} = {} [{}]".format(data_name, str(real_data.shape), str(real_data.dtype))
                for data_name, real_data in zip(data_list, data)
            ))
            print()
        return data




def load_data(configs, tokenizer, accelerator):

    """ 
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    """
    print("=====configs", configs)

    if configs["data_configs"]["dataset_name"] is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(configs["data_configs"]["dataset_name"], configs["data_configs"]["dataset_config_name"])
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                configs["data_configs"]["dataset_name"],
                configs["data_configs"]["dataset_config_name"],
                split=f"train[:{configs['data_configs']['validation_split_percentage']}%]",
            )
            raw_datasets["train"] = load_dataset(
                configs["data_configs"]["dataset_name"],
                configs["data_configs"]["dataset_config_name"],
                split=f"train[{configs['data_configs']['validation_split_percentage']}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if configs["data_configs"]["train_file"] is not None:
            data_files["train"] = configs["data_configs"]["train_file"]
        if configs["data_configs"]["validation_file"] is not None:
            data_files["validation"] = configs["data_configs"]["validation_file"]
        if configs["data_configs"]["test_file"] is not None:
            data_files["test"] = configs["data_configs"]["test_file"]
        extension = configs["data_configs"]["train_file"].split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not configs["data_configs"]["no_keep_linebreaks"]
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{configs['data_configs']['validation_split_percentage']}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{configs['data_configs']['validation_split_percentage']}%:]",
                **dataset_args,
            )


    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names ## ['text']
    text_column_name = "text" if "text" in column_names else column_names[0]




    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])



    """Datasets.map function:
    https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/main_classes#datasets.Dataset.map
    """
    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=configs["data_configs"]["preprocessing_num_workers"],
            remove_columns=column_names,
            load_from_cache_file=not configs["data_configs"]["overwrite_cache"],
            desc="Running tokenizer on dataset",
        )


    if configs["data_configs"]["block_size"] is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if configs["data_configs"]["block_size"] > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({configs['data_configs']['block_size']}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(configs["data_configs"]["block_size"], tokenizer.model_max_length)

    

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}  ### concatenated_examples.keys() =  dict_keys(['input_ids', 'attention_mask'])

        total_length = len(concatenated_examples[list(examples.keys())[0]])   # 70449
        """Example:
        concatenated_examples[:20] = [2, 2, 5457, 1738, 163, 5156, 1334, 5457, 1437, 50118, 2, 2, 1738, 163, 5156, 1334, 16, 41, 2370, 822]
        """

        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result



    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=configs["data_configs"]["preprocessing_num_workers"],
            load_from_cache_file=not configs["data_configs"]["overwrite_cache"],
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]
    test_dataset = lm_datasets["test"]


    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")



    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=configs["data_configs"]["per_device_train_batch_size"]
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=configs["data_configs"]["per_device_eval_batch_size"]
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=default_data_collator, batch_size=configs["data_configs"]["per_device_eval_batch_size"]
    )


    return train_dataloader, eval_dataloader, test_dataloader, train_dataset, eval_dataset, test_dataset
    




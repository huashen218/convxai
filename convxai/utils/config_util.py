import os
import yaml
from .utils import create_folder
from pymongo import MongoClient


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
mongodb_config_path = os.path.join(root_dir, "configs/mongodb_config.yml")
system_config_path = os.path.join(root_dir, "configs/configs.yml")


def get_mongo_connection():
    config = parse_mongodb_config_file()
    mongo = MongoClient(config['mongo_host'])[config['mongo_db_name']]
    return mongo


def parse_mongodb_config_file():
    """
    Read the config file for the MongoDB databaes settings.
    :return:
        parsed configuration dictionary
    """
    result = {}
    with open(mongodb_config_path) as f:
        cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        result["mongo_host"] = cfg.get("mongo_host")
        if not result["mongo_host"]:
            raise ValueError("Did not specify MongoDB database host")
        result["mongo_db_name"] = cfg.get("mongo_db_name")
        if not result["mongo_db_name"]:
            raise ValueError("Did not specify MongoDB database name")
    return result



def parse_system_config_file():
    """
    Read the config file for the model and data settings.
    :return:
        parsed configuration dictionary
    """
    result = {}
    with open(system_config_path) as f:
        cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)

        # result['system'] = {}
        # convxai_system = cfg.get("system")
        # for task_name, configuration in convxai_system.items():
        #     result["system"][task_name] = configuration
        #     if not result["system"][task_name]:
        #         raise ValueError(f"Did not specify '{task_name}'")
        #     if task_name == "logfilePath":
        #         create_folder([result["system"][task_name]])

        result['scientific_writing'] = {}
        scientific_writing = cfg.get("scientific_writing")
        for task_name, configuration in scientific_writing.items():
            result["scientific_writing"][task_name] = configuration
            if not result["scientific_writing"][task_name]:
                raise ValueError(f"Did not specify '{task_name}'")

        result['conversational_xai'] = {}
        conversational_xai = cfg.get("conversational_xai")
        result['conversational_xai']['checkpoints_root_dir'] = conversational_xai['checkpoints_root_dir']
        result['conversational_xai']['xai_writing_aspect_prediction_dir'] = conversational_xai['xai_writing_aspect_prediction_dir']
        result['conversational_xai']['xai_counterfactual_dir'] = conversational_xai['xai_counterfactual_dir']
        result['conversational_xai']['xai_example_dir'] = {
            'xai_emample_embeddings_dir': {},
            'xai_emample_texts_dir': {}
        }
        
        for task_name, configuration in conversational_xai['xai_example_dir']['xai_emample_embeddings_dir'].items():
            result['conversational_xai']['xai_example_dir']['xai_emample_embeddings_dir'][task_name] = configuration
            if not result['conversational_xai']['xai_example_dir']['xai_emample_embeddings_dir'][task_name]:
                raise ValueError(f"Did not specify '{task_name}'")

        for task_name, configuration in conversational_xai['xai_example_dir']['xai_emample_texts_dir'].items():
            result['conversational_xai']['xai_example_dir']['xai_emample_texts_dir'][task_name] = configuration
            if not result['conversational_xai']['xai_example_dir']['xai_emample_texts_dir'][task_name]:
                raise ValueError(f"Did not specify '{task_name}'")

    return result



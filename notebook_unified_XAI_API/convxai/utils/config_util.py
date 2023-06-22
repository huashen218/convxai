import os
import yaml
from pymongo import MongoClient


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
mongodb_config_path = os.path.join(root_dir, "configs/mongodb_config.yml")
system_config_path = os.path.join(root_dir, "configs/configs.yml")


def parse_config_file(system_config_path):
    """
    Read the config file for the model and data settings.
    
    Args:
        system_config_path: the path of the .yaml config file.

    Returns:
        The parsed configuration dictionary
    """
    with open(system_config_path) as f:
        cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return cfg



def get_mongo_connection():
    config = parse_config_file(mongodb_config_path)
    mongo = MongoClient(config['mongo_host'])[config['mongo_db_name']]
    return mongo



def parse_system_config_file():
    """
    Read the config file for the model and data settings.

    Returns:
        parsed system config dictionary
    """
    return parse_config_file(system_config_path)



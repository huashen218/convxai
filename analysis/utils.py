from pymongo import MongoClient

def get_convxai_mongo():
    return MongoClient()["convxai"]

def get_baseline_mongo():
    return MongoClient()["baseline"]
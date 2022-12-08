from pymongo import MongoClient
from bson.json_util import dumps, loads
from pathlib import Path
import os

def dump_data():
    folder = Path("data")
    os.makedirs(folder, exist_ok=True)

    # backup the convxai db
    mongo = MongoClient()["convxai"]
    data = [r for r in mongo.log.find()]
    with open(Path(folder, "convxai.json"), 'w', encoding='utf-8') as outfile:
        outfile.write(dumps(data))

    # backup the baseline db
    mongo = MongoClient()["baseline"]
    data = [r for r in mongo.log.find()]
    with open(Path(folder, "baseline.json"), 'w', encoding='utf-8') as outfile:
        outfile.write(dumps(data))

def load_data():
    with open("data/convxai.json", 'r', encoding='utf-8') as infile:
        data = loads(infile.read())
    print(data[0])

def main():
    dump_data()
    # load_data()

if __name__ == "__main__":
    main()
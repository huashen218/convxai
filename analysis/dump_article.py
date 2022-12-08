from pathlib import Path
import os
from typing import Optional
import json

from utils import get_baseline_mongo, get_convxai_mongo

def dump_text(
    mode:Optional[str] = "convxai", # (convxai, baseline)
):

    assert mode in {"convxai", "baseline"}

    if mode == "convxai":
        mongo = get_convxai_mongo()
    else:
        mongo = get_baseline_mongo()

    folder = Path("data", mode)
    os.makedirs(folder, exist_ok=True)

    for user_index in range(1, 6):
        user_id = f"P{user_index}"

        log_list = [r for r in mongo.log.find({"user_id": user_id})]
        log_list = sorted(log_list, key=lambda x: x["time"], reverse=True)

        for log in log_list:
            log["_id"] = str(log["_id"])
            log["time"] = str(log["time"])

        with open(Path(folder, f"{user_id}.json"), 'w', encoding='utf-8') as outfile:
            json.dump(log_list, outfile, indent=2)


def main():
    dump_text("convxai")
    dump_text("baseline")

if __name__ == "__main__":
    main()

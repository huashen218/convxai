from utils import get_baseline_mongo, get_convxai_mongo
from bson import ObjectId

def setup_final_writing():
    convxai_id_list = [
        "638f74848a37977af6a11c13",
        "638f9cdd8a37977af6a11c74",
        "6390c5c18c15db04700de50b",
        "6390f8758c15db04700de55a",
        "639123cd8c15db04700de5ef",
    ]

    mongo = get_convxai_mongo()
    for _id in convxai_id_list:
        mongo.log.update_one(
            {"_id": ObjectId(_id)},
            {"$set": {"final_article": True}}
        )

    baseline_id_list = [
        "638f77c4c3fa89c878d6f281",
        "638f987fc3fa89c878d6f2b3",
        "6390cb4bb41eedf55b556638",
        "6390feacb41eedf55b556678",
        "6391197fb41eedf55b5566a3",
    ]

    mongo = get_baseline_mongo()
    for _id in baseline_id_list:
        mongo.log.update_one(
            {"_id": ObjectId(_id)},
            {"$set": {"final_article": True}}
        )


def main():
    setup_final_writing()

if __name__ == "__main__":
    main()

from functools import lru_cache

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError


MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "StressSenseDB"


@lru_cache(maxsize=1)
def get_mongo_client():
    return MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000
    )


def get_database():
    return get_mongo_client()[DATABASE_NAME]


def get_users_collection():
    return get_database()["users"]


def get_history_collection():
    return get_database()["prediction_history"]


def check_database_connection():
    try:
        client = get_mongo_client()
        client.admin.command("ping")
        return True, "MongoDB connection successful!"

    except (ConnectionFailure, ServerSelectionTimeoutError) as error:
        return False, f"MongoDB connection failed: {error}"
